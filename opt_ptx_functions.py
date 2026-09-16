"""
PtX optimization models used by the global workflows.

This module defines MILP formulations for three PtX pathways:
1) DAC + PEM H2 + methanol (MeOH)
2) DAC + PEM H2 + MeOH-to-jet (MeOH->SAF)
3) DAC + PEM H2 + FT-to-jet (FT-SAF)

Core role in the project:
- Called by global scripts (e.g., 4_main_global.py, 5_main_global_grid_connected.py)
  and case-study/sensitivity scripts to size capacities and dispatch hourly operations.
- Returns techno-economic results and LCA aggregates for downstream mapping/plotting.

Key inputs (expected in df_data, length T = 8760 for hourly runs):
- pv_MW_array, wind_MW_array_on: hourly renewable availability per kWp.
- grid_abs_price: hourly grid electricity price.
- rev_inj: hourly export revenue (often zero).
- ghg_impact, ghg_impact_cons: grid GHG intensity and credit factors.
- Optional DAC fields: dac_el_kWh_per_kgCO2, dac_heat_kWh_per_kgCO2.

Key parameters:
- parm: techno-economic dictionary (capex/opex, efficiencies, lifetimes, etc.).
- dict_ghg: component LCA factors.
- dict_limits: capacity bounds for technologies.
- size_product_system: annual production target (t/yr).
- loc_elect: ISO2 code for location-specific grid factors.

Model structure (all pathways follow the same pattern):
- Decision variables for capacities (PV, wind, battery, electrolyzer, DAC, storage, process units, grid).
- Hourly dispatch for power, H2, CO2, storage, and product flows.
- Constraints for energy balances, storage dynamics, and process conversion ratios.
- Objective is annualized cost; GHG impacts are computed and returned.

Process flow scheme (energy and material pathways):

  Electricity supply:
    PV ----\
            \
    Wind -----> [Power Bus] <---- Grid import (optional)
                     |
                 [Battery] <----> (charge/discharge)
                     |
                 [PEM Electrolyzer] --> H2 storage --> H2 to synthesis
                     |
                 [DAC Unit] ---------> CO2 storage --> CO2 to synthesis
                     |
        ---------------------------------------------------------
        |                       |                              |
     MeOH synthesis        MeOH-to-jet                     FT-to-jet
        |                       |                              |
      MeOH (product)       SAF (product)                   SAF (product)

  Optional exports:
    Grid export (if enabled) and/or byproduct credits (if configured upstream).

Outputs (per pathway function):
- overview_totals: aggregated TEA + LCA metrics.
- lca_results: impact breakdown (if enabled).
- df_out: hourly operational results (optional export).

Notes/assumptions:
- Hourly resolution is standard; daily-average tests may be used upstream.
- Gurobi parameters (gap, time limit, tolerances) are configurable via kwargs.
"""

import logging
import time
from typing import Any, Dict, Tuple, Optional

import gurobipy as gp
import pandas as pd

from config import MJ_KG_H2, MJ_kWh, CC_METHOD, NAME_REF_DB

_STATUS_LABELS = {
    gp.GRB.OPTIMAL: "OPTIMAL",
    gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    gp.GRB.INFEASIBLE: "INFEASIBLE",
    gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
    gp.GRB.UNBOUNDED: "UNBOUNDED",
    gp.GRB.TIME_LIMIT: "TIME_LIMIT",
    gp.GRB.INTERRUPTED: "INTERRUPTED",
    gp.GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
}


def _model_has_solution(model: gp.Model, *, context: str, logger: bool) -> bool:
    if model.SolCount > 0:
        return True
    if logger:
        status = _STATUS_LABELS.get(model.Status, str(model.Status))
        logging.getLogger(__name__).warning("No solution for %s (status=%s).", context, status)
    return False


def _maybe_dump_iis(model: gp.Model, *, iis_path: str, logger: bool, context: str) -> None:
    if not iis_path:
        return
    try:
        model.computeIIS()
        model.write(iis_path)
        if logger:
            logging.getLogger(__name__).warning("Wrote IIS to %s for %s.", iis_path, context)
    except gp.GurobiError as exc:
        if logger:
            logging.getLogger(__name__).warning("Failed to write IIS for %s: %s", context, exc)


def _battery_overlap_metrics(
    df_out: pd.DataFrame,
    *,
    tol: float,
) -> Tuple[bool, float, int]:
    overlap = df_out[["p_battch", "p_battdis"]].min(axis=1)
    active = overlap > tol
    return bool(active.any()), float(overlap[active].sum()), int(active.sum())


def opt_dac_pem_meoh(
    df_data: pd.DataFrame,
    w_cost: float,
    w_env: float,
    parm: Dict[str, Any],
    dict_ghg: Dict[str, float],
    dict_limits: Dict[str, float],
    *,
    loc_elect: str = "GLO",
    size_product_system: float,
    credit_env_export: bool = False,
    grid_inj: bool = False,
    autonomous_elect: bool = False,
    no_renewables: bool = False,
    heuristics: bool = False,
    h2_price: float = 0.0,
    euro_ton_co2: float = 0.0,
    eps_ghg_constraint: Optional[float] = None,
    hybrid_green: bool = False,
    ghg_baseline_tco2_per_tprod: Optional[float] = None,
    ghg_reduction: float = 0.6,
    consider_down_times: bool = False,
    sec_db: str = NAME_REF_DB,
    calc_all_lca_impacts: bool = False,
    export_results: bool = False,
    export_alias: str = "",
    logger: bool = True,
    time_limit: int = 10 * 3600,
    mip_gap: float = 0.005,
    int_feas_tol: float = 1e-7,
    threads: int = 1,
    iis_path: str = "",
    battery_binary_fallback: bool = True,
    battery_overlap_tol: float = 1e-4,
    battery_overlap_energy_tol: float = 1e-3,
    battery_cycle_penalty: float = 1e-6,
    _force_battery_binary: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Optimization for DAC + PEM H2 + methanol synthesis.

    Solve a (single- or multi-objective) mixed-integer linear optimization problem
    for a DAC-PEM-methanol (MeOH) power-to-X system.

    The model co-optimizes (i) installed capacities and (ii) hourly dispatch over one
    representative year (T = 8760 h). It determines cost-optimal capacities of wind,
    solar PV, battery storage (power and energy), electrolyser, DAC (CO2 capture),
    heat pump (for DAC heat supply), H2 storage, CO2 storage, and if enabled grid
    connection, together with hourly operation variables (generation, charging/
    discharging, grid import/export, electrolysis, DAC capture, storage flows, and
    MeOH production). Renewable curtailment is represented implicitly by allowing
    renewable generation to be below its availability upper bound.

    The default objective minimizes total annual system cost (grid operating cost net
    of export revenues, annualized CAPEX, replacement costs, fixed O&M). The model can
    optionally include (i) a second objective for annual life-cycle GHG emissions
    (operational grid emissions plus annualized embodied/process emissions), (ii) a
    carbon price internalization term (€/tCO2-eq applied to the annual life-cycle GHG
    expression), and/or (iii) an ε-constraint that caps annual life-cycle GHG emissions.
    A “hybrid_green” mode can optionally enforce a relative GHG cap against a fossil
    baseline intensity.

    Args:
        df_data (pd.DataFrame): Hourly input data for the location (e.g., PV/wind
            availability, electricity prices, grid emission factors, DAC electricity/heat
            intensities, export revenues if enabled).
        w_cost (float): Weight of the cost objective (between 0 and 1).
        w_env (float): Weight of the environmental objective (between 0 and 1).
        parm (dict): Techno-economic parameters (efficiencies, lifetimes, discount rate,
            O&M fractions, process coefficients, etc.).
        dict_ghg (dict): Life-cycle GHG factors used to build the annual GHG objective
            (e.g., embodied impacts of capacities, throughput impacts for DAC/process).
        dict_limits (dict): Capacity and operational limits (upper bounds for capacities
            and dispatch variables, storage bounds, etc.).

        sec_db (str, optional): Reference database identifier for life-cycle inventory
            calculations (if LCA post-processing is enabled).
        credit_env_export (bool, optional): If True, apply environmental credits for
            exporting surplus electricity (netting export against operational emissions).
            Default is False.
        autonomous_elect (bool, optional): If True, disallow grid connection (off-grid
            autonomous system). Default is False.
        export_results (bool, optional): If True, export intermediate/final results.
            Default is False.
        eps_ghg_constraint (float or None, optional): If set, enforce an ε-constraint on
            annual life-cycle GHG emissions (tCO2-eq/yr). Default is None.
        euro_ton_co2 (float, optional): Carbon price in €/tCO2-eq applied to the annual
            life-cycle GHG expression. Default is 0.
        heuristics (bool, optional): Apply heuristic initialization/solving strategies
            (if implemented). Default is False.
        export_alias (str, optional): String appended to exported filenames. Default is "".
        mip_gap (float, optional): Relative MIP optimality gap. Default is 0.005.
        int_feas_tol (float, optional): Integrality feasibility tolerance. Default is 1e-7.
        time_limit (int, optional): Solver time limit in seconds. Default is 36,000 (10 hours).
        calc_all_lca_impacts (bool, optional): If True, calculate additional life-cycle
            impact categories beyond GHG (if implemented). Default is False.
        size_product_system (float, optional): Annual MeOH production target (t/yr).
            Default depends on your scenario setup.
        no_renewables (bool, optional): If True, exclude renewable generation from the
            system. Default is False.
        LOC_ELECT (str, optional): Geographic code for electricity data/impact factors.
            Default is 'GLO'.
        grid_inj (bool, optional): If True, allow electricity export to the grid.
            Default is False.
        hybrid_green (bool, optional): If True, enforce a relative GHG cap against a
            fossil baseline intensity. Default is False.
        ghg_baseline_tco2_per_tprod (float, optional): Fossil baseline intensity for the
            hybrid_green cap (tCO2-eq per t product). Required if hybrid_green=True.
        ghg_reduction (float, optional): Required reduction vs baseline for hybrid_green
            cap (e.g., 0.6 means 60% reduction). Default is 0.6.

    Returns:
        overview_totals (pd.DataFrame): Aggregated techno-economic and environmental results
            (cost components, capacities, annual GHG, etc.).
        lca_results (pd.DataFrame): Life-cycle assessment results by impact category (if computed).
        df_out (pd.DataFrame): Hourly operational results (dispatch, storage states, production).
    """
    start = time.time()
    delta_t = 1
    if logger:
        logging.getLogger(__name__).info("Starting PtX optimization: %s", export_alias or "MeOH")

    T = len(df_data)

    # If environmental credit for exported electricity is disabled, remove export GHG credits
    if credit_env_export is False:
        df_data = df_data.copy()
        df_data['ghg_impact_cons'] = 0
    m = gp.Model("ptx_meoh")

    if not logger:
        m.Params.OutputFlag = 0
        m.Params.LogFile = ""

    m.setParam("MIPGap", mip_gap)
    m.setParam("IntFeasTol", int_feas_tol)
    m.setParam("TimeLimit", time_limit)
    m.setParam("Presolve", 2)
    m.setParam("MIPFocus", 1)
    m.setParam("Threads", threads)
    if heuristics:
        m.setParam("NormAdjust", 2)
        m.setParam("Heuristics", 0.1)

    # Variables
    p_grid_abs = m.addVars(
        T,
        name="p_grid_abs",
        ub=0 if autonomous_elect else dict_limits["max_grid_cap"],
    )

    p_battdis = m.addVars(T, name="p_battdis", ub=dict_limits["max_bat"])
    p_battch = m.addVars(T, name="p_battch", ub=dict_limits["max_bat"])
    E_batt = m.addVars(T, name="E_batt", ub=dict_limits["max_bat"])
    # bin_bat = m.addVars(T, vtype=gp.GRB.BINARY, name="bin_bat")
    bin_bat = m.addVars(
        T,
        vtype=gp.GRB.BINARY if _force_battery_binary else gp.GRB.CONTINUOUS,
        lb=0.0,
        ub=1.0,
        name="bin_bat",
    )

    e_h2_ves = m.addVars(T, name="e_h2_ves", ub=dict_limits["max_h2_storage"])
    p_h2_ves = m.addVars(T, name="p_h2_ves", lb=-dict_limits["max_h2_storage"], ub=dict_limits["max_h2_storage"])
    e_co2_ves = m.addVars(T, name="e_co2_ves", ub=dict_limits["max_co2_storage"])
    p_co2_ves = m.addVars(T, name="p_co2_ves", lb=-dict_limits["max_co2_storage"], ub=dict_limits["max_co2_storage"])
    f_elect = m.addVars(T, name="f_elect", ub=dict_limits["st_max"])
    p_elect = m.addVars(T, name="p_elect", ub=dict_limits["st_max"])

    p_solar_pv = m.addVars(T, name="p_solar_pv", ub=dict_limits["st_max"])
    p_wind_on = m.addVars(T, name="p_wind_on", ub=dict_limits["st_max"])

    p_meoh = m.addVars(T, name="p_meoh", ub=dict_limits["st_max"])  # t MeOH/h
    p_dac = m.addVars(T, name="p_dac", ub=dict_limits["st_max"])   # t CO2/h

    if consider_down_times:
        # MeOH synthesis operational variables
        x_meoh = m.addVars(T, vtype=gp.GRB.BINARY, name="x_meoh")
        y_meoh = m.addVars(T, vtype=gp.GRB.BINARY, name="y_meoh")
        z_meoh = m.addVars(T, vtype=gp.GRB.BINARY, name="z_meoh")
        aux_meoh = m.addVars(T, name="aux_meoh", lb=0, ub=dict_limits["st_max"])

    cap_wind_on = m.addVar(name="cap_wind_on", ub=0 if no_renewables else dict_limits["max_wind_on"])
    cap_pv = m.addVar(name="cap_pv", ub=0 if no_renewables else dict_limits["max_pv"])
    cap_bat_en = m.addVar(name="cap_bat_en", ub=dict_limits["max_bat"])
    cap_bat_p = m.addVar(name="cap_bat_p", ub=dict_limits["max_bat"])
    cap_h2_ves = m.addVar(name="cap_h2_ves", ub=dict_limits["max_h2_storage"])
    cap_co2_ves = m.addVar(name="cap_co2_ves", ub=dict_limits["max_co2_storage"])
    cap_electrolyzer = m.addVar(name="cap_electrolyzer", ub=dict_limits["max_electrolyzer"])
    cap_grid = m.addVar(name="cap_grid", ub=0 if autonomous_elect else dict_limits["max_grid_cap"])
    cap_meoh = m.addVar(name="cap_meoh", ub=dict_limits["st_max"])
    cap_dac = m.addVar(name="cap_dac", ub=dict_limits["st_max"])
    cap_hp = m.addVar(name="cap_hp", ub=dict_limits.get("max_hp", dict_limits["st_max"]))  # Heat pump thermal capacity (MW_th)

    # Electricity balance
    dac_el = df_data.get("dac_el_kWh_per_kgCO2", parm.get("dac_el_kWh_per_kgCO2", 0))
    dac_heat = df_data.get("dac_heat_kWh_per_kgCO2", parm.get("dac_heat_kWh_per_kgCO2", 0))
    if not hasattr(dac_el, "__len__"):
        dac_el = [dac_el] * T
    elif hasattr(dac_el, "to_numpy"):
        dac_el = dac_el.to_numpy()
    if not hasattr(dac_heat, "__len__"):
        dac_heat = [dac_heat] * T
    elif hasattr(dac_heat, "to_numpy"):
        dac_heat = dac_heat.to_numpy()

    for t in range(T):
        m.addConstr(
            p_grid_abs[t]
            + (p_battdis[t] - p_battch[t])
            + p_solar_pv[t]
            + p_wind_on[t]
            == f_elect[t]
            + parm["meoh_el_kWh_per_kgMeOH"] * p_meoh[t]
            + dac_el[t] * p_dac[t]
            + (dac_heat[t] / parm["hp_cop"]) * p_dac[t]  # Heat pump electrical demand
            + parm.get("co2_comp_el_kWh_per_kgCO2", 0) * p_dac[t]  # CO2 compression
        )
    
    # Heat pump capacity constraint (must meet peak thermal demand)
    # dac_heat[t] [kWh/kg] × p_dac[t] [t/h] = (kWh/kg × 1000 kg/t × t/h) = 1000 kWh/h = 1000 kW = 1 MW
    for t in range(T):
        m.addConstr(dac_heat[t] * p_dac[t] <= cap_hp)  # Both in MW_th

    # H2 balance (energy basis)
    for t in range(T):
        m.addConstr(
            p_elect[t]
            == parm["meoh_kg_H2_per_kgMeOH"] * (MJ_KG_H2 / MJ_kWh / delta_t) * p_meoh[t]
            + p_h2_ves[t]
        )

    # CO2 balance: DAC supplies MeOH
    for t in range(T):
        m.addConstr(
            p_dac[t] - p_co2_ves[t] == parm["meoh_kg_CO2_per_kgMeOH"] * p_meoh[t]
        )

    # Production target
    m.addConstr(gp.quicksum(p_meoh[t] for t in range(T)) == size_product_system)

    # Power boundaries
    for t in range(T):
        m.addConstr(p_grid_abs[t] <= cap_grid)

    # Battery model
    m.addConstr(
        E_batt[0]
        == E_batt[T - 1] * (1 - parm["bat_dis_loss"] * delta_t)
        + (parm["bat_eff_ch"] * p_battch[0] * delta_t)
        - ((p_battdis[0] * delta_t) / parm["bat_eff_dis"])
    )
    for t in range(1, T):
        m.addConstr(
            E_batt[t]
            == E_batt[t - 1] * (1 - parm["bat_dis_loss"] * delta_t)
            + (parm["bat_eff_ch"] * p_battch[t] * delta_t)
            - ((p_battdis[t] * delta_t) / parm["bat_eff_dis"])
        )

    for t in range(T):
        m.addConstr(p_battch[t] <= dict_limits["max_bat"] * bin_bat[t])
        # m.addGenConstrIndicator(bin_bat[t], True, p_battdis[t], gp.GRB.EQUAL, 0)
        if _force_battery_binary:
            m.addGenConstrIndicator(bin_bat[t], True, p_battdis[t], gp.GRB.EQUAL, 0)
        else:
            m.addConstr(p_battdis[t] <= dict_limits["max_bat"] * (1 - bin_bat[t]))
        m.addConstr(p_battch[t] <= cap_bat_p)
        m.addConstr(p_battdis[t] <= cap_bat_p)
        m.addConstr(E_batt[t] >= cap_bat_en * parm["bat_soc_min"])
        m.addConstr(E_batt[t] <= cap_bat_en * parm["bat_soc_max"])

    # Hydrogen storage and electrolyzer
    for t in range(T):
        m.addConstr(p_elect[t] == f_elect[t] * parm["electr_eff"])
        m.addConstr(f_elect[t] <= cap_electrolyzer)
    m.addConstr(e_h2_ves[0] == e_h2_ves[T - 1] + p_h2_ves[0])
    for t in range(1, T):
        m.addConstr(e_h2_ves[t] == e_h2_ves[t - 1] + p_h2_ves[t])
    for t in range(T):
        m.addConstr(e_h2_ves[t] <= cap_h2_ves)
     #  m.addConstr(p_h2_ves[t] <= (cap_h2_ves / parm["h2_ramp_ves"]))
     #  m.addConstr(-(cap_h2_ves / parm["h2_ramp_ves"]) <= p_h2_ves[t])

    # CO2 storage
    m.addConstr(e_co2_ves[0] == e_co2_ves[T - 1] + p_co2_ves[0])
    for t in range(1, T):
        m.addConstr(e_co2_ves[t] == e_co2_ves[t - 1] + p_co2_ves[t])
    for t in range(T):
        m.addConstr(e_co2_ves[t] <= cap_co2_ves)
     #  m.addConstr(p_co2_ves[t] <= (cap_co2_ves / parm["co2_stor_ramp"]))
     #  m.addConstr(-(cap_co2_ves / parm["co2_stor_ramp"]) <= p_co2_ves[t])

    # Renewable availability
    for t in range(T):
        m.addConstr(p_wind_on[t] <= cap_wind_on * df_data.wind_MW_array_on.iloc[t])
        m.addConstr(p_solar_pv[t] <= cap_pv * df_data.pv_MW_array.iloc[t])

    # Process capacity limits and operational constraints (MeOH)
    if consider_down_times:
        meoh_TU = int(parm["meoh_TU"])
        meoh_TD = int(parm["meoh_TD"])
        meoh_pl_min = parm["meoh_pl_min"]
        meoh_ramp = parm["meoh_ramp"]

        for t in range(T):
            # Part-load and on/off coupling
            m.addConstr(p_meoh[t] <= aux_meoh[t])
            m.addConstr(p_meoh[t] >= meoh_pl_min * aux_meoh[t])

            # Linearization for capacity coupling
            m.addConstr(aux_meoh[t] <= dict_limits["st_max"] * x_meoh[t])
            m.addConstr(cap_meoh - dict_limits["st_max"] * (1 - x_meoh[t]) <= aux_meoh[t])
            m.addConstr(aux_meoh[t] <= cap_meoh)

            m.addConstr(p_dac[t] <= cap_dac)

        if T > 1:
            # Keep timestep 0 from exceeding timestep 1
            m.addConstr(p_meoh[0] <= p_meoh[1])

        for t in range(1, T):
            if meoh_ramp > 1:
                m.addConstr(p_meoh[t] - p_meoh[t - 1] <= (cap_meoh / meoh_ramp))
                m.addConstr(p_meoh[t - 1] - p_meoh[t] <= (cap_meoh / meoh_ramp))
            m.addConstr(x_meoh[t] - x_meoh[t - 1] == y_meoh[t] - z_meoh[t])

        if meoh_TU > 1:
            for t in range(meoh_TU, T):
                m.addConstr(
                    gp.quicksum(y_meoh[i] for i in range(t - meoh_TU + 1, t + 1)) <= x_meoh[t]
                )

        if meoh_TD > 1:
            for t in range(meoh_TD, T):
                m.addConstr(
                    gp.quicksum(z_meoh[i] for i in range(t - meoh_TD + 1, t + 1)) <= 1 - x_meoh[t]
                )

        startup_cost_term = parm.get("meoh_startup_cost", 0) * gp.quicksum(y_meoh[t] for t in range(T))
    else:
        for t in range(T):
            m.addConstr(p_meoh[t] <= cap_meoh)
            m.addConstr(p_dac[t] <= cap_dac)
        startup_cost_term = 0

    # Objective: costs
    an_op_grid_abs = gp.quicksum(p_grid_abs[t] * df_data.grid_abs_price.iloc[t] for t in range(T))
    an_op_grid_inj = 0.0
    an_op_h2_export = 0.0
    an_battery_relax_penalty = (
        0.0
        if _force_battery_binary
        else battery_cycle_penalty * gp.quicksum(p_battch[t] + p_battdis[t] for t in range(T))
    )
    an_op = startup_cost_term + an_op_grid_abs - an_op_grid_inj - an_op_h2_export + an_battery_relax_penalty

    an_capex_electrolyzer = calc_crf(parm["dr"], parm["project_lt"]) * (cap_electrolyzer * parm["electr_capex"])
    an_capex_h2_ves = calc_crf(parm["dr"], parm["project_lt"]) * (cap_h2_ves * parm["h2_ves_capex"])
    an_capex_co2_ves = calc_crf(parm["dr"], parm["project_lt"]) * (cap_co2_ves * parm["co2_stor_capex"])
    an_capex_pv = calc_crf(parm["dr"], parm["project_lt"]) * (cap_pv * parm["pv_capex"])
    an_capex_wind_on = calc_crf(parm["dr"], parm["project_lt"]) * (cap_wind_on * parm["wind_on_capex"])
    an_capex_bat_en = calc_crf(parm["dr"], parm["project_lt"]) * (cap_bat_en * parm["bat_en_capex"])
    an_capex_bat_p = calc_crf(parm["dr"], parm["project_lt"]) * (cap_bat_p * parm["bat_p_capex"])
    an_capex_grid_ins = calc_crf(parm["dr"], parm["project_lt"]) * (cap_grid * parm["grid_capex"])

    an_capex_meoh = calc_crf(parm["dr"], parm["project_lt"]) * (cap_meoh * parm["meoh_capex"])  # t/h × k€/(t/h)
    an_capex_dac = calc_crf(parm["dr"], parm["project_lt"]) * (cap_dac * parm["dac_capex"])  # t/h × k€/(t/h)
    an_capex_hp = calc_crf(parm["dr"], parm["project_lt"]) * ((cap_hp / parm["hp_cop"]) * parm["hp_capex"])  # MW_th / COP = MW_e; hp_capex is k€/MW_e

    an_capex = (
        an_capex_electrolyzer
        + an_capex_h2_ves
        + an_capex_co2_ves
        + an_capex_pv
        + an_capex_wind_on
        + an_capex_bat_en
        + an_capex_bat_p
        + an_capex_grid_ins
        + an_capex_meoh
        + an_capex_dac
        + an_capex_hp
    )

    an_rep = (
        rep_annual_int((cap_h2_ves * parm["h2_ves_capex"]), parm["project_lt"], parm["dr"], parm["h2_ves_lt"])
        + rep_annual_int((cap_co2_ves * parm["co2_stor_capex"]), parm["project_lt"], parm["dr"], parm["co2_stor_lt"])
        + rep_annual_int((cap_electrolyzer * parm["electr_capex"]), parm["project_lt"], parm["dr"], parm["electr_lt"])
        + rep_annual_int((cap_pv * parm["pv_capex"]), parm["project_lt"], parm["dr"], parm["pv_lt"])
        + rep_annual_int((cap_wind_on * parm["wind_on_capex"]), parm["project_lt"], parm["dr"], parm["wind_on_lt"])
        + rep_annual_int((cap_bat_en * parm["bat_en_capex"]), parm["project_lt"], parm["dr"], parm["bat_en_lt"])
        + rep_annual_int((cap_bat_p * parm["bat_p_capex"]), parm["project_lt"], parm["dr"], parm["bat_p_lt"])
        + rep_annual_int((cap_grid * parm["grid_capex"]), parm["project_lt"], parm["dr"], parm["grid_lt"])
        + rep_annual_int((cap_meoh * parm["meoh_capex"]), parm["project_lt"], parm["dr"], parm["meoh_lt"])  # t/h × k€/(t/h)
        + rep_annual_int((cap_dac * parm["dac_capex"]), parm["project_lt"], parm["dr"], parm["dac_lt"])  # t/h × k€/(t/h)
        + rep_annual_int(((cap_hp / parm["hp_cop"]) * parm["hp_capex"]), parm["project_lt"], parm["dr"], parm["hp_lt"])  # MW_th / COP = MW_e
    )

    an_om = (
        (cap_h2_ves * parm["h2_ves_capex"] * parm["h2_ves_om"])
        + (cap_co2_ves * parm["co2_stor_capex"] * parm["co2_stor_om"])
        + (cap_electrolyzer * parm["electr_capex"] * parm["electr_om"])
        + (cap_pv * parm["pv_capex"] * parm["pv_om"])
        + (cap_wind_on * parm["wind_on_capex"] * parm["wind_on_om"])
        + (cap_bat_en * parm["bat_en_capex"] * parm["bat_en_om"])
        + (cap_bat_p * parm["bat_p_capex"] * parm["bat_p_om"])
        + (cap_grid * parm["grid_capex"] * parm["grid_om"])
        + (cap_meoh * parm["meoh_capex"] * parm["meoh_om"])  # t/h × k€/(t/h)
        + (cap_dac * parm["dac_capex"] * parm["dac_om"])  # t/h × k€/(t/h)
        + ((cap_hp / parm["hp_cop"]) * parm["hp_capex"] * parm["hp_om"])  # MW_th / COP = MW_e
    )

    obj1_base = an_op + an_capex + an_rep + an_om

    # Objective: GHG
    an_ghg_op_grid_abs = gp.quicksum((p_grid_abs[t] * df_data.ghg_impact.iloc[t]) for t in range(T))
    an_ghg_op_grid_inj = 0.0
    an_op_ghg = an_ghg_op_grid_abs - an_ghg_op_grid_inj

    an_ghg_electrolyzer = (cap_electrolyzer * dict_ghg.get("ghg_imp_electr", 0) * (parm["project_lt"] / parm["electr_lt"])) / parm["project_lt"]
    an_ghg_h2_ves = (cap_h2_ves * dict_ghg.get("ghg_imp_h2_ves", 0) * (parm["project_lt"] / parm["h2_ves_lt"])) / parm["project_lt"]
    an_ghg_co2_ves = (cap_co2_ves * dict_ghg.get("ghg_imp_co2_ves", 0) * (parm["project_lt"] / parm["co2_stor_lt"])) / parm["project_lt"]
    an_ghg_pv = (cap_pv * dict_ghg.get("ghg_imp_pv", 0) * (parm["project_lt"] / parm["pv_lt"])) / parm["project_lt"]
    an_ghg_wind_on = (cap_wind_on * dict_ghg.get("ghg_imp_wind_on", 0) * (parm["project_lt"] / parm["wind_on_lt"])) / parm["project_lt"]
    an_ghg_bat_en = (cap_bat_en * dict_ghg.get("ghg_imp_bat_cap", 0) * (parm["project_lt"] / parm["bat_en_lt"])) / parm["project_lt"]
    an_ghg_grid_ins = (cap_grid * dict_ghg.get("ghg_impact_grid_network", 0) * (parm["project_lt"] / parm["grid_lt"])) / parm["project_lt"]
    an_ghg_hp = (cap_hp * 1000 * dict_ghg.get("ghg_imp_hp", 0) * (parm["project_lt"] / parm["hp_lt"])) / parm["project_lt"]  # MW_th to kW_th

    meoh_prod = gp.quicksum(p_meoh[t] for t in range(T))
    dac_prod = gp.quicksum(p_dac[t] for t in range(T))

    an_ghg_meoh = dict_ghg.get("ghg_imp_meoh", 0) * meoh_prod
    an_ghg_dac = dict_ghg.get("ghg_imp_dac", 0) * dac_prod

    an_op_ghg_inv = (
        an_ghg_electrolyzer
        + an_ghg_h2_ves
        + an_ghg_co2_ves
        + an_ghg_pv
        + an_ghg_wind_on
        + an_ghg_bat_en
        + an_ghg_grid_ins
        + an_ghg_hp
        + an_ghg_meoh
        + an_ghg_dac
    )
    obj2 = an_op_ghg + an_op_ghg_inv

    # Optional: internal carbon price (adds a CO2 cost proportional to lifecycle GHG emissions)
    an_op_co2 = gp.LinExpr(0)
    if euro_ton_co2 is not None and euro_ton_co2 > 0:
        # obj2 is in tCO2-eq/yr; convert €/tCO2 to k€/yr
        an_op_co2 = obj2 * (euro_ton_co2 / 1e3)

    # Final cost objective expression
    obj1 = obj1_base + an_op_co2

    # Optional: epsilon-style GHG cap (min-cost subject to annual GHG <= cap)
    ghg_cap_active = False
    if eps_ghg_constraint is not None:
        m.addConstr(obj2 <= float(eps_ghg_constraint), name="eps_ghg")
        ghg_cap_active = True

    # Optional: 'hybrid_green' cap relative to a fossil baseline (caller must provide baseline intensity)
    if hybrid_green:
        if ghg_baseline_tco2_per_tprod is None:
            raise ValueError("hybrid_green=True requires ghg_baseline_tco2_per_tprod (tCO2-eq per t product)")
        cap = float(size_product_system) * float(ghg_baseline_tco2_per_tprod) * (1.0 - float(ghg_reduction))
        m.addConstr(obj2 <= cap, name="hybrid_green_ghg")
        ghg_cap_active = True

    if ghg_cap_active:
        # Min-cost solution subject to the active GHG cap(s)
        m.setObjective(obj1)
    elif w_cost == 1 and w_env == 0:
        m.setObjective(obj1)
    elif w_cost == 0 and w_env == 1:
        m.setObjective(obj2)
    else:
        m.setObjectiveN(obj1, index=0, weight=w_cost)
        m.setObjectiveN(obj2, index=1, weight=w_env)

    try:
        m.optimize()
    except gp.GurobiError as e:
        if logger:
            print("Gurobi Error:", e)

    if not _model_has_solution(m, context=(export_alias or "MeOH"), logger=logger):
        _maybe_dump_iis(m, iis_path=iis_path, logger=logger, context=(export_alias or "MeOH"))
        return None, None, None

    # Collect results
    df_out = pd.DataFrame({
        "time": df_data.index,
        "p_grid_inj": [0] * len(df_data),
        "p_grid_abs": m.getAttr("x", p_grid_abs).values(),
        "bin_grid": [0] * len(df_data),
        "p_battdis": m.getAttr("x", p_battdis).values(),
        "p_battch": m.getAttr("x", p_battch).values(),
        "E_batt": m.getAttr("x", E_batt).values(),
        "bin_bat": m.getAttr("x", bin_bat).values(),
        "e_h2_ves": m.getAttr("x", e_h2_ves).values(),
        "p_h2_ves": m.getAttr("x", p_h2_ves).values(),
        "h2_export": [0] * len(df_data),
        "e_co2_ves": m.getAttr("x", e_co2_ves).values(),
        "p_co2_ves": m.getAttr("x", p_co2_ves).values(),
        "f_elect": m.getAttr("x", f_elect).values(),
        "p_elect": m.getAttr("x", p_elect).values(),
        "p_solar_pv": m.getAttr("x", p_solar_pv).values(),
        "p_wind_on": m.getAttr("x", p_wind_on).values(),
        "p_meoh": m.getAttr("x", p_meoh).values(),
        "p_dac": m.getAttr("x", p_dac).values(),
        "x_meoh": m.getAttr("x", x_meoh).values() if consider_down_times else [0] * len(df_data),
        "y_meoh": m.getAttr("x", y_meoh).values() if consider_down_times else [0] * len(df_data),
        "z_meoh": m.getAttr("x", z_meoh).values() if consider_down_times else [0] * len(df_data),
        "aux_meoh": m.getAttr("x", aux_meoh).values() if consider_down_times else [0] * len(df_data),
    }).set_index("time")

    if battery_binary_fallback and not _force_battery_binary:
        has_overlap, overlap_energy, overlap_hours = _battery_overlap_metrics(
            df_out, tol=battery_overlap_tol
        )
        if has_overlap and overlap_energy > battery_overlap_energy_tol:
            if logger:
                logging.getLogger(__name__).info(
                    "Battery overlap detected for %s; rerunning with binary battery logic "
                    "(hours=%s, overlap_energy=%.6f).",
                    export_alias or "MeOH",
                    overlap_hours,
                    overlap_energy,
                )
            return opt_dac_pem_meoh(
                df_data=df_data,
                w_cost=w_cost,
                w_env=w_env,
                parm=parm,
                dict_ghg=dict_ghg,
                dict_limits=dict_limits,
                loc_elect=loc_elect,
                size_product_system=size_product_system,
                credit_env_export=credit_env_export,
                grid_inj=grid_inj,
                autonomous_elect=autonomous_elect,
                no_renewables=no_renewables,
                heuristics=heuristics,
                h2_price=h2_price,
                euro_ton_co2=euro_ton_co2,
                eps_ghg_constraint=eps_ghg_constraint,
                hybrid_green=hybrid_green,
                ghg_baseline_tco2_per_tprod=ghg_baseline_tco2_per_tprod,
                ghg_reduction=ghg_reduction,
                consider_down_times=consider_down_times,
                sec_db=sec_db,
                calc_all_lca_impacts=calc_all_lca_impacts,
                export_results=export_results,
                export_alias=export_alias,
                logger=logger,
                time_limit=time_limit,
                mip_gap=mip_gap,
                int_feas_tol=int_feas_tol,
                threads=threads,
                iis_path=iis_path,
                battery_binary_fallback=battery_binary_fallback,
                battery_overlap_tol=battery_overlap_tol,
                battery_overlap_energy_tol=battery_overlap_energy_tol,
                battery_cycle_penalty=battery_cycle_penalty,
                _force_battery_binary=True,
            )

    cap_wind_on = cap_wind_on.x
    cap_pv = cap_pv.x
    cap_bat_en = cap_bat_en.x
    cap_bat_p = cap_bat_p.x
    cap_h2_ves = cap_h2_ves.x
    cap_co2_ves = cap_co2_ves.x
    cap_electrolyzer = cap_electrolyzer.x
    cap_grid = cap_grid.x
    cap_meoh = cap_meoh.x
    cap_dac = cap_dac.x
    cap_hp = cap_hp.x

    annual_costs_keuro = round(obj1.getValue(), 2)
    total_costs_raw = obj1.getValue()

    # Wall-clock runtime (hours)
    total_time = (time.time() - start) / 3600.0

    annual_ghg_emissions_kt = round(obj2.getValue() / 1e3, 2)
    total_ghg_raw = obj2.getValue()

    overview_totals = pd.DataFrame({
        "cap_pv": cap_pv,
        "cap_bat_en": cap_bat_en,
        "cap_bat_p": cap_bat_p,
        "cap_wind_on": cap_wind_on,
        "cap_h2_ves": cap_h2_ves,
        "cap_co2_ves": cap_co2_ves,
        "cap_electrolyzer": cap_electrolyzer,
        "cap_meoh": cap_meoh,
        "cap_dac": cap_dac,
        "cap_hp": cap_hp,
        "cap_grid": cap_grid,
        "total_costs": total_costs_raw,
        "operation_costs": an_op.getValue(),
        "investment_costs": an_capex.getValue(),
        "an_costs_op_co2": an_op_co2.getValue(),
        "model_status": _STATUS_LABELS.get(m.Status, str(m.Status)),
        "mip_gap": (m.MIPGap * 100 if hasattr(m, "MIPGap") else None),
        "total_time_h": total_time,
        "an_costs_op_grid_abs": an_op_grid_abs.getValue(),
        "an_costs_op_grid_inj": 0.0,
        "an_costs_capex_pv": an_capex_pv.getValue(),
        "an_costs_capex_wind_on": an_capex_wind_on.getValue(),
        "an_costs_capex_bat_en": an_capex_bat_en.getValue(),
        "an_costs_capex_bat_p": an_capex_bat_p.getValue(),
        "an_costs_capex_grid_ins": an_capex_grid_ins.getValue(),
        "an_costs_capex_electrolyzer": an_capex_electrolyzer.getValue(),
        "an_costs_capex_h2_ves": an_capex_h2_ves.getValue(),
        "an_costs_capex_co2_ves": an_capex_co2_ves.getValue(),
        "an_costs_capex_meoh": an_capex_meoh.getValue(),
        "an_costs_capex_dac": an_capex_dac.getValue(),
        "an_costs_capex_hp": an_capex_hp.getValue(),
        "an_costs_rep": an_rep.getValue(),
        "an_costs_om": an_om.getValue(),
        "tCO2_tMeOH": round(total_ghg_raw / size_product_system, 4),
        "euro_tMeOH": round(total_costs_raw * 1e3 / size_product_system, 4),
        "total_ghg": total_ghg_raw,
        "operational_ghg": an_op_ghg.getValue(),
        "an_ghg_op_grid_abs": an_ghg_op_grid_abs.getValue(),
        "an_ghg_op_grid_inj": 0.0,
        "an_ghg_pv": an_ghg_pv.getValue(),
        "an_ghg_wind_on": an_ghg_wind_on.getValue(),
        "an_ghg_bat_en": an_ghg_bat_en.getValue(),
        "an_ghg_bat_p": 0.0,
        "an_ghg_grid_ins": an_ghg_grid_ins.getValue(),
        "an_ghg_electrolyzer": an_ghg_electrolyzer.getValue(),
        "an_ghg_h2_ves": an_ghg_h2_ves.getValue(),
        "an_ghg_co2_ves": an_ghg_co2_ves.getValue(),
        "an_ghg_hp": an_ghg_hp.getValue(),
        "an_ghg_dac": an_ghg_dac.getValue(),
        "ghg_grid_mean": df_data.ghg_impact.mean(),
        "cost_grid_mean": df_data.grid_abs_price.mean(),
        "grid_elect_demand": sum(df_out.p_grid_abs),
        "battery_binary_used": int(_force_battery_binary),
    }, index=[f"opt_results_{export_alias}"])

    # Optional full LCA
    if calc_all_lca_impacts:
        import create_db_lca_functions as dnb
        lca_results = dnb.environmental_lca(
            parm=parm,
            loc_elect=loc_elect,
            cap_wind_on=cap_wind_on,
            cap_pv=cap_pv,
            cap_bat_en=cap_bat_en,
            cap_h2_ves=cap_h2_ves,
            cap_co2_ves=cap_co2_ves,
            cap_electrolyzer=cap_electrolyzer,
            cap_hb=0,
            cap_asu=0,
            cap_grid=cap_grid,
            cap_hp=cap_hp,
            summed_grid_abs=sum(df_out.p_grid_abs),
            summed_grid_inj=0.0,
            ghgs_opt=total_ghg_raw * 1000,
            w_cost=w_cost,
            sec_db=sec_db,
            credit_env_export=credit_env_export,
            lcia_method=CC_METHOD,
            epsilon_constraint=False,
            cap_dac=dac_prod.getValue() * 1e3,
            cap_meoh=meoh_prod.getValue() * 1e3,
        )
    else:
        lca_results = ""

    if export_results:
        overview_totals.T.to_excel(
            rf"results\opt_results_ptx_{export_alias}.xlsx"
        )
        df_out.to_excel(rf"results\result_ptx_{export_alias}.xlsx")

    return overview_totals, lca_results, df_out


def opt_dac_pem_meoh_to_saf(
    df_data: pd.DataFrame,
    w_cost: float,
    w_env: float,
    parm: Dict[str, Any],
    dict_ghg: Dict[str, float],
    dict_limits: Dict[str, float],
    *,
    loc_elect: str = "GLO",
    size_product_system: float,
    credit_env_export: bool = False,
    grid_inj: bool = False,
    autonomous_elect: bool = False,
    no_renewables: bool = False,
    heuristics: bool = False,
    h2_price: float = 0.0,
    euro_ton_co2: float = 0.0,
    eps_ghg_constraint: Optional[float] = None,
    saf_wtw_ghg: bool = False,
    hybrid_green: bool = False,
    ghg_baseline_tco2_per_tprod: Optional[float] = None,
    ghg_reduction: float = 0.6,
    consider_down_times: bool = False,
    sec_db: str = NAME_REF_DB,
    calc_all_lca_impacts: bool = False,
    export_results: bool = False,
    export_alias: str = "",
    logger: bool = True,
    time_limit: int = 10 * 3600,
    mip_gap: float = 0.005,
    int_feas_tol: float = 1e-7,
    threads: int = 1,
    iis_path: str = "",
    battery_binary_fallback: bool = True,
    battery_overlap_tol: float = 1e-4,
    battery_overlap_energy_tol: float = 1e-3,
    battery_cycle_penalty: float = 1e-6,
    _force_battery_binary: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Optimization for DAC + PEM H2 + MeOH + MTJ (SAF).

    Solve a (single- or multi-objective) mixed-integer linear optimization problem
    for a DAC–PEM–MeOH-to-SAF (MTJ) power-to-X system.

    The model co-optimizes installed capacities and hourly dispatch over one year
    (T = 8760 h). Decision variables include wind, solar PV, battery storage (power and
    energy), electrolyser, DAC, heat pump for DAC heat, H2 and CO2 storage, methanol
    synthesis capacity, MTJ conversion capacity, and optional grid connection. Hourly
    variables include renewable generation, battery operation, grid import/export,
    electrolysis load, DAC capture, storage flows, intermediate MeOH flow, and SAF output.
    Curtailment is represented implicitly by allowing renewable generation to be below
    availability bounds.

    The default objective minimizes total annual system cost (grid operating cost net
    of export revenues, annualized CAPEX, replacement costs, fixed O&M). The model can
    optionally include an annual life-cycle GHG objective (operational grid emissions
    plus annualized embodied/process emissions), a carbon price internalization term,
    and/or an ε-constraint on annual life-cycle GHG emissions. A “hybrid_green” option
    can enforce a relative GHG cap against a fossil SAF baseline intensity.

    Args:
        df_data (pd.DataFrame): Hourly input data (renewable availability, prices, grid
            emission factors, DAC intensities, export revenues if enabled).
        w_cost (float): Weight of the cost objective (between 0 and 1).
        w_env (float): Weight of the environmental objective (between 0 and 1).
        parm (dict): Techno-economic parameters for all technologies and process blocks.
        dict_ghg (dict): GHG factors for embodied/process contributions and grid emissions.
        dict_limits (dict): Capacity and operational limits.

        sec_db (str, optional): Reference database identifier for LCA post-processing.
        credit_env_export (bool, optional): If True, apply environmental credits for
            exported electricity. Default is False.
        autonomous_elect (bool, optional): If True, disallow grid connection. Default is False.
        export_results (bool, optional): If True, export intermediate/final results. Default is False.
        eps_ghg_constraint (float or None, optional): ε-constraint on annual life-cycle
            GHG emissions (tCO2-eq/yr). Default is None.
        euro_ton_co2 (float, optional): Carbon price (€/tCO2-eq) applied to annual life-cycle
            GHG emissions. Default is 0.
        heuristics (bool, optional): Heuristic strategies (if implemented). Default is False.
        export_alias (str, optional): Filename suffix for exports. Default is "".
        mip_gap (float, optional): Relative MIP gap. Default is 0.005.
        int_feas_tol (float, optional): Integrality feasibility tolerance. Default is 1e-7.
        time_limit (int, optional): Solver time limit in seconds. Default is 36,000.
        calc_all_lca_impacts (bool, optional): If True, compute additional impact categories
            beyond GHG (if implemented). Default is False.
        size_product_system (float, optional): Annual SAF production target (t/yr).
        no_renewables (bool, optional): If True, exclude renewables. Default is False.
        LOC_ELECT (str, optional): Electricity location code. Default is 'GLO'.
        grid_inj (bool, optional): If True, allow grid export. Default is False.
        hybrid_green (bool, optional): If True, enforce relative GHG cap vs fossil baseline.
            Default is False.
        ghg_baseline_tco2_per_tprod (float, optional): Fossil baseline intensity (tCO2-eq/t SAF)
            for hybrid_green cap. Required if hybrid_green=True.
        ghg_reduction (float, optional): Reduction vs baseline for hybrid_green cap. Default is 0.6.

    Returns:
        overview_totals (pd.DataFrame): Aggregated outputs (cost components, capacities, GHG).
        lca_results (pd.DataFrame): LCA results by impact category (if computed).
        df_out (pd.DataFrame): Hourly dispatch results including intermediate MeOH flows.
    """

    start = time.time()
    delta_t = 1
    if logger:
        logging.getLogger(__name__).info("Starting PtX optimization: %s", export_alias or "MeOH_to_SAF")

    T = len(df_data)

    # If environmental credit for exported electricity is disabled, remove export GHG credits
    if credit_env_export is False:
        df_data = df_data.copy()
        df_data['ghg_impact_cons'] = 0
    m = gp.Model("ptx_meoh_saf")

    if not logger:
        m.Params.OutputFlag = 0
        m.Params.LogFile = ""

    m.setParam("MIPGap", mip_gap)
    m.setParam("IntFeasTol", int_feas_tol)
    m.setParam("TimeLimit", time_limit)
    m.setParam("Presolve", 2)
    m.setParam("MIPFocus", 1)
    m.setParam("Threads", threads)
    if heuristics:
        m.setParam("NormAdjust", 2)
        m.setParam("Heuristics", 0.1)

    # Variables
    p_grid_abs = m.addVars(
        T,
        name="p_grid_abs",
        ub=0 if autonomous_elect else dict_limits["max_grid_cap"],
    )

    p_battdis = m.addVars(T, name="p_battdis", ub=dict_limits["max_bat"])
    p_battch = m.addVars(T, name="p_battch", ub=dict_limits["max_bat"])
    E_batt = m.addVars(T, name="E_batt", ub=dict_limits["max_bat"])
    # bin_bat = m.addVars(T, vtype=gp.GRB.BINARY, name="bin_bat")
    bin_bat = m.addVars(
        T,
        vtype=gp.GRB.BINARY if _force_battery_binary else gp.GRB.CONTINUOUS,
        lb=0.0,
        ub=1.0,
        name="bin_bat",
    )

    e_h2_ves = m.addVars(T, name="e_h2_ves", ub=dict_limits["max_h2_storage"])
    p_h2_ves = m.addVars(T, name="p_h2_ves", lb=-dict_limits["max_h2_storage"], ub=dict_limits["max_h2_storage"])
    e_co2_ves = m.addVars(T, name="e_co2_ves", ub=dict_limits["max_co2_storage"])
    p_co2_ves = m.addVars(T, name="p_co2_ves", lb=-dict_limits["max_co2_storage"], ub=dict_limits["max_co2_storage"])
    f_elect = m.addVars(T, name="f_elect", ub=dict_limits["st_max"])
    p_elect = m.addVars(T, name="p_elect", ub=dict_limits["st_max"])

    p_solar_pv = m.addVars(T, name="p_solar_pv", ub=dict_limits["st_max"])
    p_wind_on = m.addVars(T, name="p_wind_on", ub=dict_limits["st_max"])

    p_meoh = m.addVars(T, name="p_meoh", ub=dict_limits["st_max"])  # t MeOH/h
    p_saf = m.addVars(T, name="p_saf", ub=dict_limits["st_max"])    # t SAF/h
    p_dac = m.addVars(T, name="p_dac", ub=dict_limits["st_max"])    # t CO2/h

    if consider_down_times:
        # MeOH synthesis operational variables
        x_meoh = m.addVars(T, vtype=gp.GRB.BINARY, name="x_meoh")
        y_meoh = m.addVars(T, vtype=gp.GRB.BINARY, name="y_meoh")
        z_meoh = m.addVars(T, vtype=gp.GRB.BINARY, name="z_meoh")
        aux_meoh = m.addVars(T, name="aux_meoh", lb=0, ub=dict_limits["st_max"])

        # MTJ (MeOH-to-jet) operational variables
        x_mtj = m.addVars(T, vtype=gp.GRB.BINARY, name="x_mtj")
        y_mtj = m.addVars(T, vtype=gp.GRB.BINARY, name="y_mtj")
        z_mtj = m.addVars(T, vtype=gp.GRB.BINARY, name="z_mtj")
        aux_mtj = m.addVars(T, name="aux_mtj", lb=0, ub=dict_limits["st_max"])

    cap_wind_on = m.addVar(name="cap_wind_on", ub=0 if no_renewables else dict_limits["max_wind_on"])
    cap_pv = m.addVar(name="cap_pv", ub=0 if no_renewables else dict_limits["max_pv"])
    cap_bat_en = m.addVar(name="cap_bat_en", ub=dict_limits["max_bat"])
    cap_bat_p = m.addVar(name="cap_bat_p", ub=dict_limits["max_bat"])
    cap_h2_ves = m.addVar(name="cap_h2_ves", ub=dict_limits["max_h2_storage"])
    cap_co2_ves = m.addVar(name="cap_co2_ves", ub=dict_limits["max_co2_storage"])
    cap_electrolyzer = m.addVar(name="cap_electrolyzer", ub=dict_limits["max_electrolyzer"])
    cap_grid = m.addVar(name="cap_grid", ub=0 if autonomous_elect else dict_limits["max_grid_cap"])
    cap_meoh = m.addVar(name="cap_meoh", ub=dict_limits["st_max"])
    cap_mtj = m.addVar(name="cap_mtj", ub=dict_limits["st_max"])
    cap_dac = m.addVar(name="cap_dac", ub=dict_limits["st_max"])
    cap_hp = m.addVar(name="cap_hp", ub=dict_limits.get("max_hp", dict_limits["st_max"]))  # Heat pump thermal capacity (MW_th)

    dac_el = df_data.get("dac_el_kWh_per_kgCO2", parm.get("dac_el_kWh_per_kgCO2", 0))
    dac_heat = df_data.get("dac_heat_kWh_per_kgCO2", parm.get("dac_heat_kWh_per_kgCO2", 0))
    if not hasattr(dac_el, "__len__"):
        dac_el = [dac_el] * T
    elif hasattr(dac_el, "to_numpy"):
        dac_el = dac_el.to_numpy()
    if not hasattr(dac_heat, "__len__"):
        dac_heat = [dac_heat] * T
    elif hasattr(dac_heat, "to_numpy"):
        dac_heat = dac_heat.to_numpy()

    # Electricity balance
    for t in range(T):
        m.addConstr(
            p_grid_abs[t]
            + (p_battdis[t] - p_battch[t])
            + p_solar_pv[t]
            + p_wind_on[t]
            == f_elect[t]
            + parm["meoh_el_kWh_per_kgMeOH"] * p_meoh[t]
            + parm["mtj_el_kWh_per_kgSAF"] * p_saf[t]
            + dac_el[t] * p_dac[t]
            + (dac_heat[t] / parm["hp_cop"]) * p_dac[t]  # Heat pump electrical demand
            + parm.get("co2_comp_el_kWh_per_kgCO2", 0) * p_dac[t]  # CO2 compression
        )
    
    # Heat pump capacity constraint (must meet peak thermal demand)
    # dac_heat[t] [kWh/kg] × p_dac[t] [t/h] = (kWh/kg × 1000 kg/t × t/h) = 1000 kWh/h = 1000 kW = 1 MW
    for t in range(T):
        m.addConstr(dac_heat[t] * p_dac[t] <= cap_hp)  # Both in MW_th

    # H2 balance (energy basis)
    for t in range(T):
        m.addConstr(
            p_elect[t]
            == parm["meoh_kg_H2_per_kgMeOH"] * (MJ_KG_H2 / MJ_kWh / delta_t) * p_meoh[t]
            + p_h2_ves[t]
        )

    # MeOH-to-SAF conversion and CO2 balance
    for t in range(T):
        m.addConstr(p_meoh[t] == parm["mtj_kg_MeOH_per_kgSAF"] * p_saf[t])
        m.addConstr(
            p_dac[t] - p_co2_ves[t] == parm["meoh_kg_CO2_per_kgMeOH"] * p_meoh[t]
        )

    # Production target (SAF)
    m.addConstr(gp.quicksum(p_saf[t] for t in range(T)) == size_product_system)

    # Power boundaries
    for t in range(T):
        m.addConstr(p_grid_abs[t] <= cap_grid)

    # Battery model
    m.addConstr(
        E_batt[0]
        == E_batt[T - 1] * (1 - parm["bat_dis_loss"] * delta_t)
        + (parm["bat_eff_ch"] * p_battch[0] * delta_t)
        - ((p_battdis[0] * delta_t) / parm["bat_eff_dis"])
    )
    for t in range(1, T):
        m.addConstr(
            E_batt[t]
            == E_batt[t - 1] * (1 - parm["bat_dis_loss"] * delta_t)
            + (parm["bat_eff_ch"] * p_battch[t] * delta_t)
            - ((p_battdis[t] * delta_t) / parm["bat_eff_dis"])
        )

    for t in range(T):
        m.addConstr(p_battch[t] <= dict_limits["max_bat"] * bin_bat[t])
        # m.addGenConstrIndicator(bin_bat[t], True, p_battdis[t], gp.GRB.EQUAL, 0)
        if _force_battery_binary:
            m.addGenConstrIndicator(bin_bat[t], True, p_battdis[t], gp.GRB.EQUAL, 0)
        else:
            m.addConstr(p_battdis[t] <= dict_limits["max_bat"] * (1 - bin_bat[t]))
        m.addConstr(p_battch[t] <= cap_bat_p)
        m.addConstr(p_battdis[t] <= cap_bat_p)
        m.addConstr(E_batt[t] >= cap_bat_en * parm["bat_soc_min"])
        m.addConstr(E_batt[t] <= cap_bat_en * parm["bat_soc_max"])

    # Hydrogen storage and electrolyzer
    for t in range(T):
        m.addConstr(p_elect[t] == f_elect[t] * parm["electr_eff"])
        m.addConstr(f_elect[t] <= cap_electrolyzer)
    m.addConstr(e_h2_ves[0] == e_h2_ves[T - 1] + p_h2_ves[0])
    for t in range(1, T):
        m.addConstr(e_h2_ves[t] == e_h2_ves[t - 1] + p_h2_ves[t])
    for t in range(T):
        m.addConstr(e_h2_ves[t] <= cap_h2_ves)
      # m.addConstr(p_h2_ves[t] <= (cap_h2_ves / parm["h2_ramp_ves"]))
      # m.addConstr(-(cap_h2_ves / parm["h2_ramp_ves"]) <= p_h2_ves[t])

    # CO2 storage
    m.addConstr(e_co2_ves[0] == e_co2_ves[T - 1] + p_co2_ves[0])
    for t in range(1, T):
        m.addConstr(e_co2_ves[t] == e_co2_ves[t - 1] + p_co2_ves[t])
    for t in range(T):
        m.addConstr(e_co2_ves[t] <= cap_co2_ves)

    # Renewable availability
    for t in range(T):
        m.addConstr(p_wind_on[t] <= cap_wind_on * df_data.wind_MW_array_on.iloc[t])
        m.addConstr(p_solar_pv[t] <= cap_pv * df_data.pv_MW_array.iloc[t])

    # Process capacity limits and operational constraints (MeOH + MTJ)
    if consider_down_times:
        meoh_TU = int(parm["meoh_TU"])
        meoh_TD = int(parm["meoh_TD"])
        meoh_pl_min = parm["meoh_pl_min"]
        meoh_ramp = parm["meoh_ramp"]

        mtj_TU = int(parm["mtj_TU"])
        mtj_TD = int(parm["mtj_TD"])
        mtj_pl_min = parm["mtj_pl_min"]
        mtj_ramp = parm["mtj_ramp"]

        for t in range(T):
            # MeOH unit
            m.addConstr(p_meoh[t] <= aux_meoh[t])
            m.addConstr(p_meoh[t] >= meoh_pl_min * aux_meoh[t])
            m.addConstr(aux_meoh[t] <= dict_limits["st_max"] * x_meoh[t])
            m.addConstr(cap_meoh - dict_limits["st_max"] * (1 - x_meoh[t]) <= aux_meoh[t])
            m.addConstr(aux_meoh[t] <= cap_meoh)

            # MTJ unit (SAF production)
            m.addConstr(p_saf[t] <= aux_mtj[t])
            m.addConstr(p_saf[t] >= mtj_pl_min * aux_mtj[t])
            m.addConstr(aux_mtj[t] <= dict_limits["st_max"] * x_mtj[t])
            m.addConstr(cap_mtj - dict_limits["st_max"] * (1 - x_mtj[t]) <= aux_mtj[t])
            m.addConstr(aux_mtj[t] <= cap_mtj)

            m.addConstr(p_dac[t] <= cap_dac)

        if T > 1:
            m.addConstr(p_meoh[0] <= p_meoh[1])
            m.addConstr(p_saf[0] <= p_saf[1])

        for t in range(1, T):
            if meoh_ramp > 1:
                m.addConstr(p_meoh[t] - p_meoh[t - 1] <= (cap_meoh / meoh_ramp))
                m.addConstr(p_meoh[t - 1] - p_meoh[t] <= (cap_meoh / meoh_ramp))
            if mtj_ramp > 1:
                m.addConstr(p_saf[t] - p_saf[t - 1] <= (cap_mtj / mtj_ramp))
                m.addConstr(p_saf[t - 1] - p_saf[t] <= (cap_mtj / mtj_ramp))

            m.addConstr(x_meoh[t] - x_meoh[t - 1] == y_meoh[t] - z_meoh[t])
            m.addConstr(x_mtj[t] - x_mtj[t - 1] == y_mtj[t] - z_mtj[t])

        if meoh_TU > 1:
            for t in range(meoh_TU, T):
                m.addConstr(
                    gp.quicksum(y_meoh[i] for i in range(t - meoh_TU + 1, t + 1)) <= x_meoh[t]
                )
        if meoh_TD > 1:
            for t in range(meoh_TD, T):
                m.addConstr(
                    gp.quicksum(z_meoh[i] for i in range(t - meoh_TD + 1, t + 1)) <= 1 - x_meoh[t]
                )

        if mtj_TU > 1:
            for t in range(mtj_TU, T):
                m.addConstr(
                    gp.quicksum(y_mtj[i] for i in range(t - mtj_TU + 1, t + 1)) <= x_mtj[t]
                )
        if mtj_TD > 1:
            for t in range(mtj_TD, T):
                m.addConstr(
                    gp.quicksum(z_mtj[i] for i in range(t - mtj_TD + 1, t + 1)) <= 1 - x_mtj[t]
                )

        startup_cost_term = (
            parm.get("meoh_startup_cost", 0) * gp.quicksum(y_meoh[t] for t in range(T))
            + parm.get("mtj_startup_cost", 0) * gp.quicksum(y_mtj[t] for t in range(T))
        )
    else:
        for t in range(T):
            m.addConstr(p_meoh[t] <= cap_meoh)
            m.addConstr(p_saf[t] <= cap_mtj)
            m.addConstr(p_dac[t] <= cap_dac)
        startup_cost_term = 0

    # Objective: costs
    an_op_grid_abs = gp.quicksum(p_grid_abs[t] * df_data.grid_abs_price.iloc[t] for t in range(T))
    an_op_grid_inj = 0.0
    an_op_h2_export = 0.0
    an_battery_relax_penalty = (
        0.0
        if _force_battery_binary
        else battery_cycle_penalty * gp.quicksum(p_battch[t] + p_battdis[t] for t in range(T))
    )
    an_op = startup_cost_term + an_op_grid_abs - an_op_grid_inj - an_op_h2_export + an_battery_relax_penalty

    an_capex_electrolyzer = calc_crf(parm["dr"], parm["project_lt"]) * (cap_electrolyzer * parm["electr_capex"])
    an_capex_h2_ves = calc_crf(parm["dr"], parm["project_lt"]) * (cap_h2_ves * parm["h2_ves_capex"])
    an_capex_co2_ves = calc_crf(parm["dr"], parm["project_lt"]) * (cap_co2_ves * parm["co2_stor_capex"])
    an_capex_pv = calc_crf(parm["dr"], parm["project_lt"]) * (cap_pv * parm["pv_capex"])
    an_capex_wind_on = calc_crf(parm["dr"], parm["project_lt"]) * (cap_wind_on * parm["wind_on_capex"])
    an_capex_bat_en = calc_crf(parm["dr"], parm["project_lt"]) * (cap_bat_en * parm["bat_en_capex"])
    an_capex_bat_p = calc_crf(parm["dr"], parm["project_lt"]) * (cap_bat_p * parm["bat_p_capex"])
    an_capex_grid_ins = calc_crf(parm["dr"], parm["project_lt"]) * (cap_grid * parm["grid_capex"])
    an_capex_meoh = calc_crf(parm["dr"], parm["project_lt"]) * (cap_meoh * parm["meoh_capex"])  # t/h × k€/(t/h)
    an_capex_mtj = calc_crf(parm["dr"], parm["project_lt"]) * (cap_mtj * parm["mtj_capex"])  # t/h × k€/(t/h)
    an_capex_dac = calc_crf(parm["dr"], parm["project_lt"]) * (cap_dac * parm["dac_capex"])  # t/h × k€/(t/h)
    an_capex_hp = calc_crf(parm["dr"], parm["project_lt"]) * ((cap_hp / parm["hp_cop"]) * parm["hp_capex"])  # MW_th / COP = MW_e; hp_capex is k€/MW_e

    an_capex = (
        an_capex_electrolyzer
        + an_capex_h2_ves
        + an_capex_co2_ves
        + an_capex_pv
        + an_capex_wind_on
        + an_capex_bat_en
        + an_capex_bat_p
        + an_capex_grid_ins
        + an_capex_meoh
        + an_capex_mtj
        + an_capex_dac
        + an_capex_hp
    )

    an_rep = (
        rep_annual_int((cap_h2_ves * parm["h2_ves_capex"]), parm["project_lt"], parm["dr"], parm["h2_ves_lt"])
        + rep_annual_int((cap_co2_ves * parm["co2_stor_capex"]), parm["project_lt"], parm["dr"], parm["co2_stor_lt"])
        + rep_annual_int((cap_electrolyzer * parm["electr_capex"]), parm["project_lt"], parm["dr"], parm["electr_lt"])
        + rep_annual_int((cap_pv * parm["pv_capex"]), parm["project_lt"], parm["dr"], parm["pv_lt"])
        + rep_annual_int((cap_wind_on * parm["wind_on_capex"]), parm["project_lt"], parm["dr"], parm["wind_on_lt"])
        + rep_annual_int((cap_bat_en * parm["bat_en_capex"]), parm["project_lt"], parm["dr"], parm["bat_en_lt"])
        + rep_annual_int((cap_bat_p * parm["bat_p_capex"]), parm["project_lt"], parm["dr"], parm["bat_p_lt"])
        + rep_annual_int((cap_grid * parm["grid_capex"]), parm["project_lt"], parm["dr"], parm["grid_lt"])
        + rep_annual_int((cap_meoh * parm["meoh_capex"]), parm["project_lt"], parm["dr"], parm["meoh_lt"])  # t/h × k€/(t/h)
        + rep_annual_int((cap_mtj * parm["mtj_capex"]), parm["project_lt"], parm["dr"], parm["mtj_lt"])  # t/h × k€/(t/h)
        + rep_annual_int((cap_dac * parm["dac_capex"]), parm["project_lt"], parm["dr"], parm["dac_lt"])  # t/h × k€/(t/h)
        + rep_annual_int(((cap_hp / parm["hp_cop"]) * parm["hp_capex"]), parm["project_lt"], parm["dr"], parm["hp_lt"])  # MW_th / COP = MW_e
    )

    an_om = (
        (cap_h2_ves * parm["h2_ves_capex"] * parm["h2_ves_om"])
        + (cap_co2_ves * parm["co2_stor_capex"] * parm["co2_stor_om"])
        + (cap_electrolyzer * parm["electr_capex"] * parm["electr_om"])
        + (cap_pv * parm["pv_capex"] * parm["pv_om"])
        + (cap_wind_on * parm["wind_on_capex"] * parm["wind_on_om"])
        + (cap_bat_en * parm["bat_en_capex"] * parm["bat_en_om"])
        + (cap_bat_p * parm["bat_p_capex"] * parm["bat_p_om"])
        + (cap_grid * parm["grid_capex"] * parm["grid_om"])
        + (cap_meoh * parm["meoh_capex"] * parm["meoh_om"])  # t/h × k€/(t/h)
        + (cap_mtj * parm["mtj_capex"] * parm["mtj_om"])  # t/h × k€/(t/h)
        + (cap_dac * parm["dac_capex"] * parm["dac_om"])  # t/h × k€/(t/h)
        + ((cap_hp / parm["hp_cop"]) * parm["hp_capex"] * parm["hp_om"])  # MW_th / COP = MW_e
    )

    obj1_base = an_op + an_capex + an_rep + an_om

    # Objective: GHG
    an_ghg_op_grid_abs = gp.quicksum((p_grid_abs[t] * df_data.ghg_impact.iloc[t]) for t in range(T))
    an_ghg_op_grid_inj = 0.0
    an_op_ghg = an_ghg_op_grid_abs - an_ghg_op_grid_inj

    an_ghg_electrolyzer = (cap_electrolyzer * dict_ghg.get("ghg_imp_electr", 0) * (parm["project_lt"] / parm["electr_lt"])) / parm["project_lt"]
    an_ghg_h2_ves = (cap_h2_ves * dict_ghg.get("ghg_imp_h2_ves", 0) * (parm["project_lt"] / parm["h2_ves_lt"])) / parm["project_lt"]
    an_ghg_co2_ves = (cap_co2_ves * dict_ghg.get("ghg_imp_co2_ves", 0) * (parm["project_lt"] / parm["co2_stor_lt"])) / parm["project_lt"]
    an_ghg_pv = (cap_pv * dict_ghg.get("ghg_imp_pv", 0) * (parm["project_lt"] / parm["pv_lt"])) / parm["project_lt"]
    an_ghg_wind_on = (cap_wind_on * dict_ghg.get("ghg_imp_wind_on", 0) * (parm["project_lt"] / parm["wind_on_lt"])) / parm["project_lt"]
    an_ghg_bat_en = (cap_bat_en * dict_ghg.get("ghg_imp_bat_cap", 0) * (parm["project_lt"] / parm["bat_en_lt"])) / parm["project_lt"]
    an_ghg_grid_ins = (cap_grid * dict_ghg.get("ghg_impact_grid_network", 0) * (parm["project_lt"] / parm["grid_lt"])) / parm["project_lt"]
    an_ghg_hp = (cap_hp * 1000 * dict_ghg.get("ghg_imp_hp", 0) * (parm["project_lt"] / parm["hp_lt"])) / parm["project_lt"]  # MW_th to kW_th

    saf_prod = gp.quicksum(p_saf[t] for t in range(T))
    meoh_prod = gp.quicksum(p_meoh[t] for t in range(T))
    dac_prod = gp.quicksum(p_dac[t] for t in range(T))

    an_ghg_mtj = dict_ghg.get("ghg_imp_mtj", 0) * saf_prod
    an_ghg_meoh = dict_ghg.get("ghg_imp_meoh", 0) * meoh_prod
    an_ghg_dac = dict_ghg.get("ghg_imp_dac", 0) * dac_prod

    an_op_ghg_inv = (
        an_ghg_electrolyzer
        + an_ghg_h2_ves
        + an_ghg_co2_ves
        + an_ghg_pv
        + an_ghg_wind_on
        + an_ghg_bat_en
        + an_ghg_grid_ins
        + an_ghg_hp
        + an_ghg_meoh
        + an_ghg_mtj
        + an_ghg_dac
    )
    obj2 = an_op_ghg + an_op_ghg_inv
    ghg_metric = obj2 - an_ghg_dac if saf_wtw_ghg else obj2

    # Optional: internal carbon price (adds a CO2 cost proportional to lifecycle GHG emissions)
    an_op_co2 = gp.LinExpr(0)
    if euro_ton_co2 is not None and euro_ton_co2 > 0:
        # ghg_metric is in tCO2-eq/yr; convert €/tCO2 to k€/yr
        an_op_co2 = ghg_metric * (euro_ton_co2 / 1e3)

    # Final cost objective expression
    obj1 = obj1_base + an_op_co2

    # Optional: epsilon-style GHG cap (min-cost subject to annual GHG <= cap)
    ghg_cap_active = False
    if eps_ghg_constraint is not None:
        m.addConstr(ghg_metric <= float(eps_ghg_constraint), name="eps_ghg")
        ghg_cap_active = True

    # Optional: 'hybrid_green' cap relative to a fossil baseline (caller must provide baseline intensity)
    if hybrid_green:
        if ghg_baseline_tco2_per_tprod is None:
            raise ValueError("hybrid_green=True requires ghg_baseline_tco2_per_tprod (tCO2-eq per t product)")
        cap = float(size_product_system) * float(ghg_baseline_tco2_per_tprod) * (1.0 - float(ghg_reduction))
        m.addConstr(ghg_metric <= cap, name="hybrid_green_ghg")
        ghg_cap_active = True

    if ghg_cap_active:
        # Min-cost solution subject to the active GHG cap(s)
        m.setObjective(obj1)
    elif w_cost == 1 and w_env == 0:
        m.setObjective(obj1)
    elif w_cost == 0 and w_env == 1:
        m.setObjective(ghg_metric)
    else:
        m.setObjectiveN(obj1, index=0, weight=w_cost)
        m.setObjectiveN(ghg_metric, index=1, weight=w_env)

    try:
        m.optimize()
    except gp.GurobiError as e:
        if logger:
            print("Gurobi Error:", e)

    if not _model_has_solution(m, context=(export_alias or "MeOH_to_SAF"), logger=logger):
        _maybe_dump_iis(m, iis_path=iis_path, logger=logger, context=(export_alias or "MeOH_to_SAF"))
        return None, None, None

    df_out = pd.DataFrame({
        "time": df_data.index,
        "p_grid_inj": [0] * len(df_data),
        "p_grid_abs": m.getAttr("x", p_grid_abs).values(),
        "bin_grid": [0] * len(df_data),
        "p_battdis": m.getAttr("x", p_battdis).values(),
        "p_battch": m.getAttr("x", p_battch).values(),
        "E_batt": m.getAttr("x", E_batt).values(),
        "bin_bat": m.getAttr("x", bin_bat).values(),
        "e_h2_ves": m.getAttr("x", e_h2_ves).values(),
        "p_h2_ves": m.getAttr("x", p_h2_ves).values(),
        "e_co2_ves": m.getAttr("x", e_co2_ves).values(),
        "p_co2_ves": m.getAttr("x", p_co2_ves).values(),
        "f_elect": m.getAttr("x", f_elect).values(),
        "p_elect": m.getAttr("x", p_elect).values(),
        "p_solar_pv": m.getAttr("x", p_solar_pv).values(),
        "p_wind_on": m.getAttr("x", p_wind_on).values(),
        "p_meoh": m.getAttr("x", p_meoh).values(),
        "p_saf": m.getAttr("x", p_saf).values(),
        "p_dac": m.getAttr("x", p_dac).values(),
        "x_meoh": m.getAttr("x", x_meoh).values() if consider_down_times else [0] * len(df_data),
        "y_meoh": m.getAttr("x", y_meoh).values() if consider_down_times else [0] * len(df_data),
        "z_meoh": m.getAttr("x", z_meoh).values() if consider_down_times else [0] * len(df_data),
        "aux_meoh": m.getAttr("x", aux_meoh).values() if consider_down_times else [0] * len(df_data),
        "x_mtj": m.getAttr("x", x_mtj).values() if consider_down_times else [0] * len(df_data),
        "y_mtj": m.getAttr("x", y_mtj).values() if consider_down_times else [0] * len(df_data),
        "z_mtj": m.getAttr("x", z_mtj).values() if consider_down_times else [0] * len(df_data),
        "aux_mtj": m.getAttr("x", aux_mtj).values() if consider_down_times else [0] * len(df_data),
    }).set_index("time")

    if battery_binary_fallback and not _force_battery_binary:
        has_overlap, overlap_energy, overlap_hours = _battery_overlap_metrics(
            df_out, tol=battery_overlap_tol
        )
        if has_overlap and overlap_energy > battery_overlap_energy_tol:
            if logger:
                logging.getLogger(__name__).info(
                    "Battery overlap detected for %s; rerunning with binary battery logic "
                    "(hours=%s, overlap_energy=%.6f).",
                    export_alias or "MeOH_to_SAF",
                    overlap_hours,
                    overlap_energy,
                )
            return opt_dac_pem_meoh_to_saf(
                df_data=df_data,
                w_cost=w_cost,
                w_env=w_env,
                parm=parm,
                dict_ghg=dict_ghg,
                dict_limits=dict_limits,
                loc_elect=loc_elect,
                size_product_system=size_product_system,
                credit_env_export=credit_env_export,
                grid_inj=grid_inj,
                autonomous_elect=autonomous_elect,
                no_renewables=no_renewables,
                heuristics=heuristics,
                h2_price=h2_price,
                euro_ton_co2=euro_ton_co2,
                eps_ghg_constraint=eps_ghg_constraint,
                hybrid_green=hybrid_green,
                ghg_baseline_tco2_per_tprod=ghg_baseline_tco2_per_tprod,
                ghg_reduction=ghg_reduction,
                consider_down_times=consider_down_times,
                sec_db=sec_db,
                calc_all_lca_impacts=calc_all_lca_impacts,
                export_results=export_results,
                export_alias=export_alias,
                logger=logger,
                time_limit=time_limit,
                mip_gap=mip_gap,
                int_feas_tol=int_feas_tol,
                threads=threads,
                iis_path=iis_path,
                battery_binary_fallback=battery_binary_fallback,
                battery_overlap_tol=battery_overlap_tol,
                battery_overlap_energy_tol=battery_overlap_energy_tol,
                battery_cycle_penalty=battery_cycle_penalty,
                _force_battery_binary=True,
            )

    cap_wind_on = cap_wind_on.x
    cap_pv = cap_pv.x
    cap_bat_en = cap_bat_en.x
    cap_bat_p = cap_bat_p.x
    cap_h2_ves = cap_h2_ves.x
    cap_co2_ves = cap_co2_ves.x
    cap_electrolyzer = cap_electrolyzer.x
    cap_grid = cap_grid.x
    cap_meoh = cap_meoh.x
    cap_mtj = cap_mtj.x
    cap_dac = cap_dac.x
    cap_hp = cap_hp.x

    annual_costs_keuro = round(obj1.getValue(), 2)
    total_costs_raw = obj1.getValue()

    # Wall-clock runtime (hours)
    total_time = (time.time() - start) / 3600.0

    annual_ghg_emissions_kt = round(ghg_metric.getValue() / 1e3, 2)
    total_ghg_raw = obj2.getValue()
    total_ghg_metric = ghg_metric.getValue()

    overview_totals = pd.DataFrame({
        "cap_pv": cap_pv,
        "cap_bat_en": cap_bat_en,
        "cap_bat_p": cap_bat_p,
        "cap_wind_on": cap_wind_on,
        "cap_h2_ves": cap_h2_ves,
        "cap_co2_ves": cap_co2_ves,
        "cap_electrolyzer": cap_electrolyzer,
        "cap_meoh": cap_meoh,
        "cap_mtj": cap_mtj,
        "cap_dac": cap_dac,
        "cap_hp": cap_hp,
        "cap_grid": cap_grid,
        "total_costs": total_costs_raw,
        "operation_costs": an_op.getValue(),
        "investment_costs": an_capex.getValue(),
        "an_costs_op_co2": an_op_co2.getValue(),
        "model_status": _STATUS_LABELS.get(m.Status, str(m.Status)),
        "mip_gap": (m.MIPGap * 100 if hasattr(m, "MIPGap") else None),
        "total_time_h": total_time,
        "an_costs_op_grid_abs": an_op_grid_abs.getValue(),
        "an_costs_op_grid_inj": 0.0,
        "an_costs_capex_pv": an_capex_pv.getValue(),
        "an_costs_capex_wind_on": an_capex_wind_on.getValue(),
        "an_costs_capex_bat_en": an_capex_bat_en.getValue(),
        "an_costs_capex_bat_p": an_capex_bat_p.getValue(),
        "an_costs_capex_grid_ins": an_capex_grid_ins.getValue(),
        "an_costs_capex_electrolyzer": an_capex_electrolyzer.getValue(),
        "an_costs_capex_h2_ves": an_capex_h2_ves.getValue(),
        "an_costs_capex_co2_ves": an_capex_co2_ves.getValue(),
        "an_costs_capex_meoh": an_capex_meoh.getValue(),
        "an_costs_capex_mtj": an_capex_mtj.getValue(),
        "an_costs_capex_dac": an_capex_dac.getValue(),
        "an_costs_capex_hp": an_capex_hp.getValue(),
        "an_costs_rep": an_rep.getValue(),
        "an_costs_om": an_om.getValue(),
        "tCO2_tSAF": round(total_ghg_metric / size_product_system, 4),
        "tCO2_tSAF_raw": round(total_ghg_raw / size_product_system, 4),
        "euro_tSAF": round(total_costs_raw * 1e3 / size_product_system, 4),
        "total_ghg": total_ghg_metric,
        "total_ghg_raw": total_ghg_raw,
        "operational_ghg": an_op_ghg.getValue(),
        "an_ghg_op_grid_abs": an_ghg_op_grid_abs.getValue(),
        "an_ghg_op_grid_inj": 0.0,
        "an_ghg_pv": an_ghg_pv.getValue(),
        "an_ghg_wind_on": an_ghg_wind_on.getValue(),
        "an_ghg_bat_en": an_ghg_bat_en.getValue(),
        "an_ghg_bat_p": 0.0,
        "an_ghg_grid_ins": an_ghg_grid_ins.getValue(),
        "an_ghg_electrolyzer": an_ghg_electrolyzer.getValue(),
        "an_ghg_h2_ves": an_ghg_h2_ves.getValue(),
        "an_ghg_co2_ves": an_ghg_co2_ves.getValue(),
        "an_ghg_hp": an_ghg_hp.getValue(),
        "an_ghg_dac": an_ghg_dac.getValue(),
        "ghg_basis": "wtw_no_dac_credit" if saf_wtw_ghg else "net_with_dac_credit",
        "ghg_grid_mean": df_data.ghg_impact.mean(),
        "cost_grid_mean": df_data.grid_abs_price.mean(),
        "grid_elect_demand": sum(df_out.p_grid_abs),
        "battery_binary_used": int(_force_battery_binary),
    }, index=[f"opt_results_{export_alias}"])

    if calc_all_lca_impacts:
        import create_db_lca_functions as dnb
        lca_results = dnb.environmental_lca(
            parm=parm,
            loc_elect=loc_elect,
            cap_wind_on=cap_wind_on,
            cap_pv=cap_pv,
            cap_bat_en=cap_bat_en,
            cap_h2_ves=cap_h2_ves,
            cap_co2_ves=cap_co2_ves,
            cap_electrolyzer=cap_electrolyzer,
            cap_hb=0,
            cap_asu=0,
            cap_grid=cap_grid,
            cap_hp=cap_hp,
            summed_grid_abs=sum(df_out.p_grid_abs),
            summed_grid_inj=0.0,
            ghgs_opt=total_ghg_raw * 1000,
            w_cost=w_cost,
            sec_db=sec_db,
            credit_env_export=credit_env_export,
            lcia_method=CC_METHOD,
            epsilon_constraint=False,
            cap_dac=dac_prod.getValue() * 1e3,
            cap_meoh=meoh_prod.getValue() * 1e3,
            cap_mtj=saf_prod.getValue() * 1e3,
        )
    else:
        lca_results = ""

    if export_results:
        overview_totals.T.to_excel(
            rf"results\opt_results_ptx_{export_alias}.xlsx"
        )
        df_out.to_excel(rf"results\result_ptx_{export_alias}.xlsx")

    return overview_totals, lca_results, df_out


def opt_dac_pem_ftsaf(
    df_data: pd.DataFrame,
    w_cost: float,
    w_env: float,
    parm: Dict[str, Any],
    dict_ghg: Dict[str, float],
    dict_limits: Dict[str, float],
    *,
    loc_elect: str = "GLO",
    size_product_system: float,
    credit_env_export: bool = False,
    grid_inj: bool = False,
    autonomous_elect: bool = False,
    no_renewables: bool = False,
    heuristics: bool = False,
    h2_price: float = 0.0,
    euro_ton_co2: float = 0.0,
    eps_ghg_constraint: Optional[float] = None,
    saf_wtw_ghg: bool = False,
    hybrid_green: bool = False,
    ghg_baseline_tco2_per_tprod: Optional[float] = None,
    ghg_reduction: float = 0.6,
    consider_down_times: bool = False,
    sec_db: str = NAME_REF_DB,
    calc_all_lca_impacts: bool = False,
    export_results: bool = False,
    export_alias: str = "",
    logger: bool = True,
    time_limit: int = 10 * 3600,
    mip_gap: float = 0.005,
    int_feas_tol: float = 1e-7,
    threads: int = 1,
    iis_path: str = "",
    battery_binary_fallback: bool = True,
    battery_overlap_tol: float = 1e-4,
    battery_overlap_energy_tol: float = 1e-3,
    battery_cycle_penalty: float = 1e-6,
    _force_battery_binary: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Optimization for DAC + PEM H2 + FT-to-jet SAF (black-box).
    """
    start = time.time()
    delta_t = 1
    if logger:
        logging.getLogger(__name__).info("Starting PtX optimization: %s", export_alias or "FT_to_SAF")

    T = len(df_data)

    # If environmental credit for exported electricity is disabled, remove export GHG credits
    if credit_env_export is False:
        df_data = df_data.copy()
        df_data['ghg_impact_cons'] = 0
    m = gp.Model("ptx_ftsaf")

    if not logger:
        m.Params.OutputFlag = 0
        m.Params.LogFile = ""

    m.setParam("MIPGap", mip_gap)
    m.setParam("IntFeasTol", int_feas_tol)
    m.setParam("TimeLimit", time_limit)
    m.setParam("Presolve", 2)
    m.setParam("MIPFocus", 1)
    m.setParam("Threads", threads)
    if heuristics:
        m.setParam("NormAdjust", 2)
        m.setParam("Heuristics", 0.1)

    # Variables
    p_grid_abs = m.addVars(
        T,
        name="p_grid_abs",
        ub=0 if autonomous_elect else dict_limits["max_grid_cap"],
    )

    p_battdis = m.addVars(T, name="p_battdis", ub=dict_limits["max_bat"])
    p_battch = m.addVars(T, name="p_battch", ub=dict_limits["max_bat"])
    E_batt = m.addVars(T, name="E_batt", ub=dict_limits["max_bat"])
    # bin_bat = m.addVars(T, vtype=gp.GRB.BINARY, name="bin_bat")
    bin_bat = m.addVars(
        T,
        vtype=gp.GRB.BINARY if _force_battery_binary else gp.GRB.CONTINUOUS,
        lb=0.0,
        ub=1.0,
        name="bin_bat",
    )

    e_h2_ves = m.addVars(T, name="e_h2_ves", ub=dict_limits["max_h2_storage"])
    p_h2_ves = m.addVars(T, name="p_h2_ves", lb=-dict_limits["max_h2_storage"], ub=dict_limits["max_h2_storage"])
    e_co2_ves = m.addVars(T, name="e_co2_ves", ub=dict_limits["max_co2_storage"])
    p_co2_ves = m.addVars(T, name="p_co2_ves", lb=-dict_limits["max_co2_storage"], ub=dict_limits["max_co2_storage"])
    f_elect = m.addVars(T, name="f_elect", ub=dict_limits["st_max"])
    p_elect = m.addVars(T, name="p_elect", ub=dict_limits["st_max"])

    p_solar_pv = m.addVars(T, name="p_solar_pv", ub=dict_limits["st_max"])
    p_wind_on = m.addVars(T, name="p_wind_on", ub=dict_limits["st_max"])

    p_saf = m.addVars(T, name="p_saf", ub=dict_limits["st_max"])    # t SAF/h
    p_dac = m.addVars(T, name="p_dac", ub=dict_limits["st_max"])    # t CO2/h

    if consider_down_times:
        # FT-SAF operational variables
        x_ftsaf = m.addVars(T, vtype=gp.GRB.BINARY, name="x_ftsaf")
        y_ftsaf = m.addVars(T, vtype=gp.GRB.BINARY, name="y_ftsaf")
        z_ftsaf = m.addVars(T, vtype=gp.GRB.BINARY, name="z_ftsaf")
        aux_ftsaf = m.addVars(T, name="aux_ftsaf", lb=0, ub=dict_limits["st_max"])

    cap_wind_on = m.addVar(name="cap_wind_on", ub=0 if no_renewables else dict_limits["max_wind_on"])
    cap_pv = m.addVar(name="cap_pv", ub=0 if no_renewables else dict_limits["max_pv"])
    cap_bat_en = m.addVar(name="cap_bat_en", ub=dict_limits["max_bat"])
    cap_bat_p = m.addVar(name="cap_bat_p", ub=dict_limits["max_bat"])
    cap_h2_ves = m.addVar(name="cap_h2_ves", ub=dict_limits["max_h2_storage"])
    cap_co2_ves = m.addVar(name="cap_co2_ves", ub=dict_limits["max_co2_storage"])
    cap_electrolyzer = m.addVar(name="cap_electrolyzer", ub=dict_limits["max_electrolyzer"])
    cap_grid = m.addVar(name="cap_grid", ub=0 if autonomous_elect else dict_limits["max_grid_cap"])
    cap_ftsaf = m.addVar(name="cap_ftsaf", ub=dict_limits["st_max"])
    cap_dac = m.addVar(name="cap_dac", ub=dict_limits["st_max"])
    cap_hp = m.addVar(name="cap_hp", ub=dict_limits.get("max_hp", dict_limits["st_max"]))  # Heat pump thermal capacity (MW_th)

    dac_el = df_data.get("dac_el_kWh_per_kgCO2", parm.get("dac_el_kWh_per_kgCO2", 0))
    dac_heat = df_data.get("dac_heat_kWh_per_kgCO2", parm.get("dac_heat_kWh_per_kgCO2", 0))
    if not hasattr(dac_el, "__len__"):
        dac_el = [dac_el] * T
    elif hasattr(dac_el, "to_numpy"):
        dac_el = dac_el.to_numpy()
    if not hasattr(dac_heat, "__len__"):
        dac_heat = [dac_heat] * T
    elif hasattr(dac_heat, "to_numpy"):
        dac_heat = dac_heat.to_numpy()

    # Electricity balance
    for t in range(T):
        m.addConstr(
            p_grid_abs[t]
            + (p_battdis[t] - p_battch[t])
            + p_solar_pv[t]
            + p_wind_on[t]
            == f_elect[t]
            + parm["ftsaf_el_kWh_per_kgSAF"] * p_saf[t]
            + dac_el[t] * p_dac[t]
            + (dac_heat[t] / parm["hp_cop"]) * p_dac[t]  # Heat pump electrical demand
            + parm.get("co2_comp_el_kWh_per_kgCO2", 0) * p_dac[t]  # CO2 compression
        )
    
    # Heat pump capacity constraint (must meet peak thermal demand)
    # dac_heat[t] [kWh/kg] × p_dac[t] [t/h] = (kWh/kg × 1000 kg/t × t/h) = 1000 kWh/h = 1000 kW = 1 MW
    for t in range(T):
        m.addConstr(dac_heat[t] * p_dac[t] <= cap_hp)  # Both in MW_th

    # H2 balance (energy basis)
    for t in range(T):
        m.addConstr(
            p_elect[t]
            == parm["ftsaf_kg_H2_per_kgSAF_FT"] * (MJ_KG_H2 / MJ_kWh / delta_t) * p_saf[t]
            + p_h2_ves[t]
        )

    # CO2 balance
    for t in range(T):
        m.addConstr(
            p_dac[t] - p_co2_ves[t] == parm["ftsaf_kg_CO2_per_kgSAF_FT"] * p_saf[t]
        )

    # Production target (SAF)
    m.addConstr(gp.quicksum(p_saf[t] for t in range(T)) == size_product_system)

    # Power boundaries
    for t in range(T):
        m.addConstr(p_grid_abs[t] <= cap_grid)

    # Battery model
    m.addConstr(
        E_batt[0]
        == E_batt[T - 1] * (1 - parm["bat_dis_loss"] * delta_t)
        + (parm["bat_eff_ch"] * p_battch[0] * delta_t)
        - ((p_battdis[0] * delta_t) / parm["bat_eff_dis"])
    )
    for t in range(1, T):
        m.addConstr(
            E_batt[t]
            == E_batt[t - 1] * (1 - parm["bat_dis_loss"] * delta_t)
            + (parm["bat_eff_ch"] * p_battch[t] * delta_t)
            - ((p_battdis[t] * delta_t) / parm["bat_eff_dis"])
        )

    for t in range(T):
        m.addConstr(p_battch[t] <= dict_limits["max_bat"] * bin_bat[t])
        # m.addGenConstrIndicator(bin_bat[t], True, p_battdis[t], gp.GRB.EQUAL, 0)
        if _force_battery_binary:
            m.addGenConstrIndicator(bin_bat[t], True, p_battdis[t], gp.GRB.EQUAL, 0)
        else:
            m.addConstr(p_battdis[t] <= dict_limits["max_bat"] * (1 - bin_bat[t]))
        m.addConstr(p_battch[t] <= cap_bat_p)
        m.addConstr(p_battdis[t] <= cap_bat_p)
        m.addConstr(E_batt[t] >= cap_bat_en * parm["bat_soc_min"])
        m.addConstr(E_batt[t] <= cap_bat_en * parm["bat_soc_max"])

    # Hydrogen storage and electrolyzer
    for t in range(T):
        m.addConstr(p_elect[t] == f_elect[t] * parm["electr_eff"])
        m.addConstr(f_elect[t] <= cap_electrolyzer)
    m.addConstr(e_h2_ves[0] == e_h2_ves[T - 1] + p_h2_ves[0])
    for t in range(1, T):
        m.addConstr(e_h2_ves[t] == e_h2_ves[t - 1] + p_h2_ves[t])
    for t in range(T):
        m.addConstr(e_h2_ves[t] <= cap_h2_ves)
     #  m.addConstr(p_h2_ves[t] <= (cap_h2_ves / parm["h2_ramp_ves"]))
     #  m.addConstr(-(cap_h2_ves / parm["h2_ramp_ves"]) <= p_h2_ves[t])

    # CO2 storage
    m.addConstr(e_co2_ves[0] == e_co2_ves[T - 1] + p_co2_ves[0])
    for t in range(1, T):
        m.addConstr(e_co2_ves[t] == e_co2_ves[t - 1] + p_co2_ves[t])
    for t in range(T):
        m.addConstr(e_co2_ves[t] <= cap_co2_ves)
      # m.addConstr(p_co2_ves[t] <= (cap_co2_ves / parm["co2_stor_ramp"]))
      # m.addConstr(-(cap_co2_ves / parm["co2_stor_ramp"]) <= p_co2_ves[t])

    # Renewable availability
    for t in range(T):
        m.addConstr(p_wind_on[t] <= cap_wind_on * df_data.wind_MW_array_on.iloc[t])
        m.addConstr(p_solar_pv[t] <= cap_pv * df_data.pv_MW_array.iloc[t])

    # Process capacity limits and operational constraints (FT-SAF)
    if consider_down_times:
        ftsaf_TU = int(parm["ftsaf_TU"])
        ftsaf_TD = int(parm["ftsaf_TD"])
        ftsaf_pl_min = parm["ftsaf_pl_min"]
        ftsaf_ramp = parm["ftsaf_ramp"]

        for t in range(T):
            m.addConstr(p_saf[t] <= aux_ftsaf[t])
            m.addConstr(p_saf[t] >= ftsaf_pl_min * aux_ftsaf[t])
            m.addConstr(aux_ftsaf[t] <= dict_limits["st_max"] * x_ftsaf[t])
            m.addConstr(cap_ftsaf - dict_limits["st_max"] * (1 - x_ftsaf[t]) <= aux_ftsaf[t])
            m.addConstr(aux_ftsaf[t] <= cap_ftsaf)
            m.addConstr(p_dac[t] <= cap_dac)

        if T > 1:
            m.addConstr(p_saf[0] <= p_saf[1])

        for t in range(1, T):
            if ftsaf_ramp > 1:
                m.addConstr(p_saf[t] - p_saf[t - 1] <= (cap_ftsaf / ftsaf_ramp))
                m.addConstr(p_saf[t - 1] - p_saf[t] <= (cap_ftsaf / ftsaf_ramp))
            m.addConstr(x_ftsaf[t] - x_ftsaf[t - 1] == y_ftsaf[t] - z_ftsaf[t])

        if ftsaf_TU > 1:
            for t in range(ftsaf_TU, T):
                m.addConstr(
                    gp.quicksum(y_ftsaf[i] for i in range(t - ftsaf_TU + 1, t + 1)) <= x_ftsaf[t]
                )
        if ftsaf_TD > 1:
            for t in range(ftsaf_TD, T):
                m.addConstr(
                    gp.quicksum(z_ftsaf[i] for i in range(t - ftsaf_TD + 1, t + 1)) <= 1 - x_ftsaf[t]
                )

        startup_cost_term = parm.get("ftsaf_startup_cost", 0) * gp.quicksum(y_ftsaf[t] for t in range(T))
    else:
        for t in range(T):
            m.addConstr(p_saf[t] <= cap_ftsaf)
            m.addConstr(p_dac[t] <= cap_dac)
        startup_cost_term = 0

    # Objective: costs
    an_op_grid_abs = gp.quicksum(p_grid_abs[t] * df_data.grid_abs_price.iloc[t] for t in range(T))
    an_op_grid_inj = 0.0
    an_op_h2_export = 0.0
    an_battery_relax_penalty = (
        0.0
        if _force_battery_binary
        else battery_cycle_penalty * gp.quicksum(p_battch[t] + p_battdis[t] for t in range(T))
    )
    an_op = startup_cost_term + an_op_grid_abs - an_op_grid_inj - an_op_h2_export + an_battery_relax_penalty

    an_capex_electrolyzer = calc_crf(parm["dr"], parm["project_lt"]) * (cap_electrolyzer * parm["electr_capex"])
    an_capex_h2_ves = calc_crf(parm["dr"], parm["project_lt"]) * (cap_h2_ves * parm["h2_ves_capex"])
    an_capex_co2_ves = calc_crf(parm["dr"], parm["project_lt"]) * (cap_co2_ves * parm["co2_stor_capex"])
    an_capex_pv = calc_crf(parm["dr"], parm["project_lt"]) * (cap_pv * parm["pv_capex"])
    an_capex_wind_on = calc_crf(parm["dr"], parm["project_lt"]) * (cap_wind_on * parm["wind_on_capex"])
    an_capex_bat_en = calc_crf(parm["dr"], parm["project_lt"]) * (cap_bat_en * parm["bat_en_capex"])
    an_capex_bat_p = calc_crf(parm["dr"], parm["project_lt"]) * (cap_bat_p * parm["bat_p_capex"])
    an_capex_grid_ins = calc_crf(parm["dr"], parm["project_lt"]) * (cap_grid * parm["grid_capex"])
    an_capex_ftsaf = calc_crf(parm["dr"], parm["project_lt"]) * (cap_ftsaf * parm["ftsaf_capex"])  # t/h × k€/(t/h)
    an_capex_dac = calc_crf(parm["dr"], parm["project_lt"]) * (cap_dac * parm["dac_capex"])  # t/h × k€/(t/h)
    an_capex_hp = calc_crf(parm["dr"], parm["project_lt"]) * ((cap_hp / parm["hp_cop"]) * parm["hp_capex"])  # MW_th / COP = MW_e; hp_capex is k€/MW_e

    an_capex = (
        an_capex_electrolyzer
        + an_capex_h2_ves
        + an_capex_co2_ves
        + an_capex_pv
        + an_capex_wind_on
        + an_capex_bat_en
        + an_capex_bat_p
        + an_capex_grid_ins
        + an_capex_ftsaf
        + an_capex_dac
        + an_capex_hp
    )

    an_rep = (
        rep_annual_int((cap_h2_ves * parm["h2_ves_capex"]), parm["project_lt"], parm["dr"], parm["h2_ves_lt"])
        + rep_annual_int((cap_co2_ves * parm["co2_stor_capex"]), parm["project_lt"], parm["dr"], parm["co2_stor_lt"])
        + rep_annual_int((cap_electrolyzer * parm["electr_capex"]), parm["project_lt"], parm["dr"], parm["electr_lt"])
        + rep_annual_int((cap_pv * parm["pv_capex"]), parm["project_lt"], parm["dr"], parm["pv_lt"])
        + rep_annual_int((cap_wind_on * parm["wind_on_capex"]), parm["project_lt"], parm["dr"], parm["wind_on_lt"])
        + rep_annual_int((cap_bat_en * parm["bat_en_capex"]), parm["project_lt"], parm["dr"], parm["bat_en_lt"])
        + rep_annual_int((cap_bat_p * parm["bat_p_capex"]), parm["project_lt"], parm["dr"], parm["bat_p_lt"])
        + rep_annual_int((cap_grid * parm["grid_capex"]), parm["project_lt"], parm["dr"], parm["grid_lt"])
        + rep_annual_int((cap_ftsaf * parm["ftsaf_capex"]), parm["project_lt"], parm["dr"], parm["ftsaf_lt"])  # t/h × k€/(t/h)
        + rep_annual_int((cap_dac * parm["dac_capex"]), parm["project_lt"], parm["dr"], parm["dac_lt"])  # t/h × k€/(t/h)
        + rep_annual_int(((cap_hp / parm["hp_cop"]) * parm["hp_capex"]), parm["project_lt"], parm["dr"], parm["hp_lt"])  # MW_th / COP = MW_e
    )

    an_om = (
        (cap_h2_ves * parm["h2_ves_capex"] * parm["h2_ves_om"])
        + (cap_co2_ves * parm["co2_stor_capex"] * parm["co2_stor_om"])
        + (cap_electrolyzer * parm["electr_capex"] * parm["electr_om"])
        + (cap_pv * parm["pv_capex"] * parm["pv_om"])
        + (cap_wind_on * parm["wind_on_capex"] * parm["wind_on_om"])
        + (cap_bat_en * parm["bat_en_capex"] * parm["bat_en_om"])
        + (cap_bat_p * parm["bat_p_capex"] * parm["bat_p_om"])
        + (cap_grid * parm["grid_capex"] * parm["grid_om"])
        + (cap_ftsaf * parm["ftsaf_capex"] * parm["ftsaf_om"])  # t/h × k€/(t/h)
        + (cap_dac * parm["dac_capex"] * parm["dac_om"])  # t/h × k€/(t/h)
        + ((cap_hp / parm["hp_cop"]) * parm["hp_capex"] * parm["hp_om"])  # MW_th / COP = MW_e
    )

    obj1_base = an_op + an_capex + an_rep + an_om

    # Objective: GHG
    an_ghg_op_grid_abs = gp.quicksum((p_grid_abs[t] * df_data.ghg_impact.iloc[t]) for t in range(T))
    an_ghg_op_grid_inj = 0.0
    an_op_ghg = an_ghg_op_grid_abs - an_ghg_op_grid_inj

    an_ghg_electrolyzer = (cap_electrolyzer * dict_ghg.get("ghg_imp_electr", 0) * (parm["project_lt"] / parm["electr_lt"])) / parm["project_lt"]
    an_ghg_h2_ves = (cap_h2_ves * dict_ghg.get("ghg_imp_h2_ves", 0) * (parm["project_lt"] / parm["h2_ves_lt"])) / parm["project_lt"]
    an_ghg_co2_ves = (cap_co2_ves * dict_ghg.get("ghg_imp_co2_ves", 0) * (parm["project_lt"] / parm["co2_stor_lt"])) / parm["project_lt"]
    an_ghg_pv = (cap_pv * dict_ghg.get("ghg_imp_pv", 0) * (parm["project_lt"] / parm["pv_lt"])) / parm["project_lt"]
    an_ghg_wind_on = (cap_wind_on * dict_ghg.get("ghg_imp_wind_on", 0) * (parm["project_lt"] / parm["wind_on_lt"])) / parm["project_lt"]
    an_ghg_bat_en = (cap_bat_en * dict_ghg.get("ghg_imp_bat_cap", 0) * (parm["project_lt"] / parm["bat_en_lt"])) / parm["project_lt"]
    an_ghg_grid_ins = (cap_grid * dict_ghg.get("ghg_impact_grid_network", 0) * (parm["project_lt"] / parm["grid_lt"])) / parm["project_lt"]
    an_ghg_hp = (cap_hp * 1000 * dict_ghg.get("ghg_imp_hp", 0) * (parm["project_lt"] / parm["hp_lt"])) / parm["project_lt"]  # MW_th to kW_th

    saf_prod = gp.quicksum(p_saf[t] for t in range(T))
    dac_prod = gp.quicksum(p_dac[t] for t in range(T))

    an_ghg_ftsaf = dict_ghg.get("ghg_imp_ftsaf", 0) * saf_prod
    an_ghg_dac = dict_ghg.get("ghg_imp_dac", 0) * dac_prod

    an_op_ghg_inv = (
        an_ghg_electrolyzer
        + an_ghg_h2_ves
        + an_ghg_co2_ves
        + an_ghg_pv
        + an_ghg_wind_on
        + an_ghg_bat_en
        + an_ghg_grid_ins
        + an_ghg_hp
        + an_ghg_ftsaf
        + an_ghg_dac
    )
    obj2 = an_op_ghg + an_op_ghg_inv
    ghg_metric = obj2 - an_ghg_dac if saf_wtw_ghg else obj2

    # Optional: internal carbon price (adds a CO2 cost proportional to lifecycle GHG emissions)
    an_op_co2 = gp.LinExpr(0)
    if euro_ton_co2 is not None and euro_ton_co2 > 0:
        # ghg_metric is in tCO2-eq/yr; convert €/tCO2 to k€/yr
        an_op_co2 = ghg_metric * (euro_ton_co2 / 1e3)

    # Final cost objective expression
    obj1 = obj1_base + an_op_co2

    # Optional: epsilon-style GHG cap (min-cost subject to annual GHG <= cap)
    ghg_cap_active = False
    if eps_ghg_constraint is not None:
        m.addConstr(ghg_metric <= float(eps_ghg_constraint), name="eps_ghg")
        ghg_cap_active = True

    # Optional: 'hybrid_green' cap relative to a fossil baseline (caller must provide baseline intensity)
    if hybrid_green:
        if ghg_baseline_tco2_per_tprod is None:
            raise ValueError("hybrid_green=True requires ghg_baseline_tco2_per_tprod (tCO2-eq per t product)")
        cap = float(size_product_system) * float(ghg_baseline_tco2_per_tprod) * (1.0 - float(ghg_reduction))
        m.addConstr(ghg_metric <= cap, name="hybrid_green_ghg")
        ghg_cap_active = True

    if ghg_cap_active:
        # Min-cost solution subject to the active GHG cap(s)
        m.setObjective(obj1)
    elif w_cost == 1 and w_env == 0:
        m.setObjective(obj1)
    elif w_cost == 0 and w_env == 1:
        m.setObjective(ghg_metric)
    else:
        m.setObjectiveN(obj1, index=0, weight=w_cost)
        m.setObjectiveN(ghg_metric, index=1, weight=w_env)

    try:
        m.optimize()
    except gp.GurobiError as e:
        if logger:
            print("Gurobi Error:", e)

    if not _model_has_solution(m, context=(export_alias or "FT_to_SAF"), logger=logger):
        _maybe_dump_iis(m, iis_path=iis_path, logger=logger, context=(export_alias or "FT_to_SAF"))
        return None, None, None

    df_out = pd.DataFrame({
        "time": df_data.index,
        "p_grid_inj": [0] * len(df_data),
        "p_grid_abs": m.getAttr("x", p_grid_abs).values(),
        "bin_grid": [0] * len(df_data),
        "p_battdis": m.getAttr("x", p_battdis).values(),
        "p_battch": m.getAttr("x", p_battch).values(),
        "E_batt": m.getAttr("x", E_batt).values(),
        "bin_bat": m.getAttr("x", bin_bat).values(),
        "e_h2_ves": m.getAttr("x", e_h2_ves).values(),
        "p_h2_ves": m.getAttr("x", p_h2_ves).values(),
        "e_co2_ves": m.getAttr("x", e_co2_ves).values(),
        "p_co2_ves": m.getAttr("x", p_co2_ves).values(),
        "f_elect": m.getAttr("x", f_elect).values(),
        "p_elect": m.getAttr("x", p_elect).values(),
        "p_solar_pv": m.getAttr("x", p_solar_pv).values(),
        "p_wind_on": m.getAttr("x", p_wind_on).values(),
        "p_saf": m.getAttr("x", p_saf).values(),
        "p_dac": m.getAttr("x", p_dac).values(),
        "x_ftsaf": m.getAttr("x", x_ftsaf).values() if consider_down_times else [0] * len(df_data),
        "y_ftsaf": m.getAttr("x", y_ftsaf).values() if consider_down_times else [0] * len(df_data),
        "z_ftsaf": m.getAttr("x", z_ftsaf).values() if consider_down_times else [0] * len(df_data),
        "aux_ftsaf": m.getAttr("x", aux_ftsaf).values() if consider_down_times else [0] * len(df_data),
    }).set_index("time")

    if battery_binary_fallback and not _force_battery_binary:
        has_overlap, overlap_energy, overlap_hours = _battery_overlap_metrics(
            df_out, tol=battery_overlap_tol
        )
        if has_overlap and overlap_energy > battery_overlap_energy_tol:
            if logger:
                logging.getLogger(__name__).info(
                    "Battery overlap detected for %s; rerunning with binary battery logic "
                    "(hours=%s, overlap_energy=%.6f).",
                    export_alias or "FT_to_SAF",
                    overlap_hours,
                    overlap_energy,
                )
            return opt_dac_pem_ftsaf(
                df_data=df_data,
                w_cost=w_cost,
                w_env=w_env,
                parm=parm,
                dict_ghg=dict_ghg,
                dict_limits=dict_limits,
                loc_elect=loc_elect,
                size_product_system=size_product_system,
                credit_env_export=credit_env_export,
                grid_inj=grid_inj,
                autonomous_elect=autonomous_elect,
                no_renewables=no_renewables,
                heuristics=heuristics,
                h2_price=h2_price,
                euro_ton_co2=euro_ton_co2,
                eps_ghg_constraint=eps_ghg_constraint,
                hybrid_green=hybrid_green,
                ghg_baseline_tco2_per_tprod=ghg_baseline_tco2_per_tprod,
                ghg_reduction=ghg_reduction,
                consider_down_times=consider_down_times,
                sec_db=sec_db,
                calc_all_lca_impacts=calc_all_lca_impacts,
                export_results=export_results,
                export_alias=export_alias,
                logger=logger,
                time_limit=time_limit,
                mip_gap=mip_gap,
                int_feas_tol=int_feas_tol,
                threads=threads,
                iis_path=iis_path,
                battery_binary_fallback=battery_binary_fallback,
                battery_overlap_tol=battery_overlap_tol,
                battery_overlap_energy_tol=battery_overlap_energy_tol,
                battery_cycle_penalty=battery_cycle_penalty,
                _force_battery_binary=True,
            )

    cap_wind_on = cap_wind_on.x
    cap_pv = cap_pv.x
    cap_bat_en = cap_bat_en.x
    cap_bat_p = cap_bat_p.x
    cap_h2_ves = cap_h2_ves.x
    cap_co2_ves = cap_co2_ves.x
    cap_electrolyzer = cap_electrolyzer.x
    cap_grid = cap_grid.x
    cap_ftsaf = cap_ftsaf.x
    cap_dac = cap_dac.x
    cap_hp = cap_hp.x

    annual_costs_keuro = round(obj1.getValue(), 2)
    total_costs_raw = obj1.getValue()

    # Wall-clock runtime (hours)
    total_time = (time.time() - start) / 3600.0

    annual_ghg_emissions_kt = round(ghg_metric.getValue() / 1e3, 2)
    total_ghg_raw = obj2.getValue()
    total_ghg_metric = ghg_metric.getValue()

    overview_totals = pd.DataFrame({
        "cap_pv": cap_pv,
        "cap_bat_en": cap_bat_en,
        "cap_bat_p": cap_bat_p,
        "cap_wind_on": cap_wind_on,
        "cap_h2_ves": cap_h2_ves,
        "cap_co2_ves": cap_co2_ves,
        "cap_electrolyzer": cap_electrolyzer,
        "cap_ftsaf": cap_ftsaf,
        "cap_dac": cap_dac,
        "cap_hp": cap_hp,
        "cap_grid": cap_grid,
        "total_costs": total_costs_raw,
        "operation_costs": an_op.getValue(),
        "investment_costs": an_capex.getValue(),
        "an_costs_op_co2": an_op_co2.getValue(),
        "model_status": _STATUS_LABELS.get(m.Status, str(m.Status)),
        "mip_gap": (m.MIPGap * 100 if hasattr(m, "MIPGap") else None),
        "total_time_h": total_time,
        "an_costs_op_grid_abs": an_op_grid_abs.getValue(),
        "an_costs_op_grid_inj": 0.0,
        "an_costs_capex_pv": an_capex_pv.getValue(),
        "an_costs_capex_wind_on": an_capex_wind_on.getValue(),
        "an_costs_capex_bat_en": an_capex_bat_en.getValue(),
        "an_costs_capex_bat_p": an_capex_bat_p.getValue(),
        "an_costs_capex_grid_ins": an_capex_grid_ins.getValue(),
        "an_costs_capex_electrolyzer": an_capex_electrolyzer.getValue(),
        "an_costs_capex_h2_ves": an_capex_h2_ves.getValue(),
        "an_costs_capex_co2_ves": an_capex_co2_ves.getValue(),
        "an_costs_capex_ftsaf": an_capex_ftsaf.getValue(),
        "an_costs_capex_dac": an_capex_dac.getValue(),
        "an_costs_capex_hp": an_capex_hp.getValue(),
        "an_costs_rep": an_rep.getValue(),
        "an_costs_om": an_om.getValue(),
        "tCO2_tSAF": round(total_ghg_metric / size_product_system, 4),
        "tCO2_tSAF_raw": round(total_ghg_raw / size_product_system, 4),
        "euro_tSAF": round(total_costs_raw * 1e3 / size_product_system, 4),
        "total_ghg": total_ghg_metric,
        "total_ghg_raw": total_ghg_raw,
        "operational_ghg": an_op_ghg.getValue(),
        "an_ghg_op_grid_abs": an_ghg_op_grid_abs.getValue(),
        "an_ghg_op_grid_inj": 0.0,
        "an_ghg_pv": an_ghg_pv.getValue(),
        "an_ghg_wind_on": an_ghg_wind_on.getValue(),
        "an_ghg_bat_en": an_ghg_bat_en.getValue(),
        "an_ghg_bat_p": 0.0,
        "an_ghg_grid_ins": an_ghg_grid_ins.getValue(),
        "an_ghg_electrolyzer": an_ghg_electrolyzer.getValue(),
        "an_ghg_h2_ves": an_ghg_h2_ves.getValue(),
        "an_ghg_co2_ves": an_ghg_co2_ves.getValue(),
        "an_ghg_hp": an_ghg_hp.getValue(),
        "an_ghg_dac": an_ghg_dac.getValue(),
        "ghg_basis": "wtw_no_dac_credit" if saf_wtw_ghg else "net_with_dac_credit",
        "ghg_grid_mean": df_data.ghg_impact.mean(),
        "cost_grid_mean": df_data.grid_abs_price.mean(),
        "grid_elect_demand": sum(df_out.p_grid_abs),
        "battery_binary_used": int(_force_battery_binary),
    }, index=[f"opt_results_{export_alias}"])

    if calc_all_lca_impacts:
        import create_db_lca_functions as dnb
        lca_results = dnb.environmental_lca(
            parm=parm,
            loc_elect=loc_elect,
            cap_wind_on=cap_wind_on,
            cap_pv=cap_pv,
            cap_bat_en=cap_bat_en,
            cap_h2_ves=cap_h2_ves,
            cap_co2_ves=cap_co2_ves,
            cap_electrolyzer=cap_electrolyzer,
            cap_hb=0,
            cap_asu=0,
            cap_grid=cap_grid,
            cap_hp=cap_hp,
            summed_grid_abs=sum(df_out.p_grid_abs),
            summed_grid_inj=0.0,
            ghgs_opt=total_ghg_raw * 1000,
            w_cost=w_cost,
            sec_db=sec_db,
            credit_env_export=credit_env_export,
            lcia_method=CC_METHOD,
            epsilon_constraint=False,
            cap_dac=dac_prod.getValue() * 1e3,
            cap_ftsaf=saf_prod.getValue() * 1e3,
        )
    else:
        lca_results = ""

    if export_results:
        overview_totals.T.to_excel(
            rf"results\opt_results_ptx_{export_alias}.xlsx"
        )
        df_out.to_excel(rf"results\result_ptx_{export_alias}.xlsx")

    return overview_totals, lca_results, df_out

def calc_crf(r: float, lt: float) -> float:
    return (r * (1 + r) ** lt) / ((1 + r) ** lt - 1)


def rep_annual_int(unit_inv_cost: float, lt: int, r: float, ry: float, rep_factor=0.75) -> float:
    c_rep_total = 0
    unit_inv_cost = rep_factor * unit_inv_cost
    if ry > lt:
        rest = 1 - (lt / ry)
        crf = calc_crf(r, lt)
        c_rep_a = crf * ((unit_inv_cost * rest) / ((1 + r) ** lt))
        c_rep_total += c_rep_a
    else:
        for i in range(int(lt / ry)):
            c_rep_total += (unit_inv_cost / (1 + r) ** (ry * (i + 1)))
    crf = calc_crf(r, lt)
    return crf * c_rep_total


def calc_meoh_to_saf_block(annual_meoh_t: float, parm: Dict[str, Any]) -> Dict[str, float]:
    """
    Black-box MeOH-to-SAF block (MTJ).

    Returns annual SAF output, electricity demand, and CAPEX/O&M placeholders
    based on parameter inputs.
    """
    saf_t = annual_meoh_t / parm["kg_MeOH_per_kgSAF"]
    el_kwh = parm["mtj_el_kWh_per_kgSAF"] * saf_t * 1000
    capex = parm["mtj_capex"] * (saf_t / 8760)
    return {"saf_t": saf_t, "el_kwh": el_kwh, "capex": capex}


def calc_ftsaf_block(annual_saf_t: float, parm: Dict[str, Any]) -> Dict[str, float]:
    """
    Black-box FT-to-SAF block.
    """
    el_kwh = parm["ftsaf_el_kWh_per_kgSAF"] * annual_saf_t * 1000
    capex = parm["ftsaf_capex"] * (annual_saf_t / 8760)
    return {"saf_t": annual_saf_t, "el_kwh": el_kwh, "capex": capex}
