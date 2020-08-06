import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import pandapower as pp
import pandapower.plotting as plot
import pandapower.networks as nw
import simbench as sb

from matplotlib.pyplot import get_cmap
from matplotlib.colors import Normalize
import pandapower.plotting.plotly as pplotly
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import vlevel_plotly
from pandapower.plotting.plotly import pf_res_plotly
from pandas import Series

# sb_code1 = "1-LV-urban6--0-sw"
sb_code1 = "1-MVLV-urban-5.303-0-sw"
# sb_code1 = "1-LV-rural1--0-sw"
net = sb.get_simbench_net(sb_code1)
# net = nw.example_multivoltage()
# net = nw.mv_oberrhein()

profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

# profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
# net.trafo.tap_pos = 1


time_steps = range(0, 700)
results = pd.DataFrame([], index=time_steps, columns=["PV_power", "line_loading", 'voltage_pu_max',
                                                      'voltage_pu_min', 'Consumer', 'ext_grid_Power', 'line_mean'])
results_flex = pd.DataFrame([], index=time_steps, columns=["PV_power", "line_loading", 'voltage_pu_max',
                                                           'voltage_pu_min', 'Consumer', 'ext_grid_Power', 'line_mean'])


def apply_absolute_values(net, absolute_values_dict, case_or_time_step):
    for elm_param in absolute_values_dict.keys():
        if absolute_values_dict[elm_param].shape[1]:
            elm = elm_param[0]
            param = elm_param[1]
            net[elm].loc[:, param] = absolute_values_dict[elm_param].loc[case_or_time_step]


def apply_flexibility(net, load_id, power, start_time, end_time, type, time_step):
    if start_time <= time_step <= end_time:
        pow_flex = net[type].loc[net[type]['name'] == load_id, 'p_mw']
        net[type].loc[net[type]['name'] == load_id, 'p_mw'] = pow_flex + power


for time_step in time_steps:
    apply_absolute_values(net, profiles, time_step)
    net['sgen'].loc[:, 'p_mw'] = net['sgen'].loc[:, 'p_mw'] * 5
    pp.runpp(net)
    results.loc[time_step, "PV_power"] = net.res_sgen.p_mw.sum()
    # results.loc[time_step, "line_loading"] = net.res_line.loading_percent[5]
    results.loc[time_step, "line_loading"] = net.res_line.loading_percent.max()
    results.loc[time_step, "voltage_pu_max"] = net.res_bus.vm_pu.max()
    results.loc[time_step, "voltage_pu_min"] = net.res_bus.vm_pu.min()
    results.loc[time_step, "Consumer"] = net.res_load.p_mw.sum()
    results.loc[time_step, "ext_grid_Power"] = net.res_ext_grid['p_mw'][0]
    results.loc[time_step, "line_mean"] = net.res_line.loading_percent.mean()

    # apply_flexibility(net, 'LV5.303 Load 13', 0/1000, 100, 400, 'load', time_step)
    apply_flexibility(net, 'MV3.101 Load 74', 1200/1000, 100, 250, 'load', time_step)
    apply_flexibility(net, 'MV3.101 Load 75', 1500/1000, 350, 600, 'load', time_step)
    # apply_flexibility(net, 'MV3.101 SGen 5', 500 / 1000, 100, 400, 'sgen', time_step)

    pp.runpp(net)
    results_flex.loc[time_step, "PV_power"] = net.res_sgen.p_mw.sum()
    # results.loc[time_step, "line_loading"] = net.res_line.loading_percent[5]
    results_flex.loc[time_step, "line_loading"] = net.res_line.loading_percent.max()
    results_flex.loc[time_step, "voltage_pu_max"] = net.res_bus.vm_pu.max()
    results_flex.loc[time_step, "voltage_pu_min"] = net.res_bus.vm_pu.min()
    results_flex.loc[time_step, "Consumer"] = net.res_load.p_mw.sum()
    results_flex.loc[time_step, "ext_grid_Power"] = net.res_ext_grid['p_mw'][0]
    results_flex.loc[time_step, "line_mean"] = net.res_line.loading_percent.mean()

# costeg = pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=10)
# costgen1 = pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=10)
# costgen2 = pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=10)


# pp.runopp(net)
# pp.runpp(net)
# simple_plotly(net)
pf_res_plotly(net)
# vlevel_plotly(net)

# change the load of bus 27/ load 23, Type G1-A
# net['load'].loc[36, 'p_mw']

plt.figure(0)
plt.subplot(2, 2, 1)
plt.plot(results.loc[:, "PV_power"])
plt.plot(results_flex.loc[:, "PV_power"])
plt.title('RES_Power')
plt.xlabel('timesteps [15 min]')
plt.ylabel('Power [MW]')
plt.legend(['ref', 'flex'])

plt.subplot(2, 2, 2)
plt.plot(results.loc[:, "line_loading"])
plt.plot(results_flex.loc[:, "line_loading"])
plt.title('Line_loading')
plt.xlabel('timesteps [15 min ]')
plt.ylabel('Line_loading in percent [%]')
plt.legend(['ref', 'flex'])

plt.subplot(2, 2, 3)
plt.plot(results.loc[:, "voltage_pu_max"])
plt.plot(results.loc[:, "voltage_pu_min"])
plt.plot(results_flex.loc[:, "voltage_pu_max"])
plt.plot(results_flex.loc[:, "voltage_pu_min"])
plt.title('voltage_range', y=-0.01)
plt.xlabel('timesteps [15 min ]')
plt.ylabel('Voltage per unit [pu]')
plt.legend(['voltage_pu_max', 'voltage_pu_min', 'voltage_pu_max_flex', 'voltage_pu_min_flex'])

plt.subplot(2, 2, 4)
plt.plot(results.loc[:, "Consumer"], '--')
plt.plot(results_flex.loc[:, "Consumer"], '--')
plt.title('load commercial/household', y=-0.01)
plt.xlabel('timesteps [15 min ]')
plt.ylabel('Power [MW]')
plt.legend(['ref', 'flex'])

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(results.loc[:, "ext_grid_Power"])
plt.plot(results_flex.loc[:, "ext_grid_Power"])
plt.title('ext_grid_Power')
plt.xlabel('timesteps [15 min]')
plt.ylabel('Power [MW]')
plt.legend(['ref', 'flex'])

plt.subplot(2, 1, 2)
plt.plot(results.loc[:, "line_mean"])
plt.plot(results_flex.loc[:, "line_mean"])
plt.title('line_loading_mean', y=-0.01)
plt.xlabel('timesteps [15 min]')
plt.ylabel('line_loading [%]')
plt.legend(['ref', 'flex'])
