--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

gd = require "gd"
require "gnuplot"

if not dqn then
    require "initenv"
end

require 'cutorch'
require 'cunn'

print("vai")


local game = 'galaga'
local networkstring = 'DQN3_0_1_' .. game .. '_FULL_Y.t7'
local network

local msg, err = pcall(require, networkstring)
-- try to load saved agent
local err_msg, exp = pcall(torch.load, networkstring)
if not err_msg then
error("Could not find network file. Error: " .. exp)
end
print("Loading network from file...")
network = exp

properties = {network.time_history, network.reward_history, network.reward_counts, network.episode_counts, network.v_history, network.td_history, network.qmax_history}

labels = {"time (s)", "Max_Reward", "Reward_Counts", "Episode_Counts", "V_History", "td_history", "qmax_History"}

table.remove(properties[1], #properties[1])

for i=1, #properties, 1 do
	print('size of ' .. i .. ' is ' .. #properties[i])
end

for i=2, #properties, 1 do
	gnuplot.pngfigure("../saves/" .. game .."/" .. labels[i] .. '.png')
	gnuplot.plot(torch.Tensor(properties[1]), torch.Tensor(properties[i]))
	gnuplot.xlabel(labels[1])
	gnuplot.ylabel(labels[i])
	gnuplot.plotflush()
end

-- if exp.best_model then
--     print("Loading best model...")
--     print(exp)
--     network = exp.best_model
-- else
--     print("Loading the latest (not necessarily the best) model...")
--     network = exp.model
-- end
