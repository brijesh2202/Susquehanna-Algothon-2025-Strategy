from os import wait3

import pandas as pd
from pandas import DataFrame, Series
from typing import TypedDict, List, Dict, Any, Union
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
from importlib.machinery import ModuleSpec
from types import ModuleType, FunctionType
from matplotlib.axes import Axes
import json

# CONSTANTS #######################################################################################
START_DAY: int = 0
END_DAY: int = 0
INSTRUMENT_POSITION_LIMIT: int = 10000
COMMISSION_RATE: float = 0.0005
NUMBER_OF_INSTRUMENTS: int = 50

PLOT_COLORS: Dict[str, str] = {
    "pnl": "#2ca02c",
    "cum_pnl": "#1f77b4",
    "utilisation": "#ff7f0e",
    "sharpe_change": "#d62728",
}

default_strategy_filepath: str = "./Main.py"
default_strategy_function_name: str = "getMyPosition"
strategy_file_not_found_message: str = "Strategy file not found"
could_not_load_spec_message: str = "Could not load spec for module from strategy file"
strategy_function_does_not_exist_message: str = (
    "getMyPosition function does not exist in strategy " "file"
)
strategy_function_not_callable_message: str = "getMyPosition function is not callable"

usage_error: str = """
    Usage: backtester.py [OPTIONS]
    (Usage instructions as provided by you)
"""

CMD_LINE_OPTIONS: List[str] = [
    "--path",
    "--function-name",
    "--timeline",
    "--disable-comms",
    "--show",
]

GRAPH_OPTIONS: List[str] = [
    "daily-pnl",
    "cum-pnl",
    "capital-util",
    "sharpe-heat-map",
    "cum-sharpe",
]


# TYPE DECLARATIONS ###############################################################################
class InstrumentPriceEntry(TypedDict):
    day: int
    instrument: int
    price: float


class Trade(TypedDict):
    price_entry: float
    order_type: str
    day: int


class BacktesterResults(TypedDict):
    daily_pnl: ndarray
    daily_capital_utilisation: ndarray
    daily_instrument_returns: ndarray
    trades: Dict[int, List[Trade]]
    start_day: int
    end_day: int


class Params:
    def __init__(
        self,
        strategy_filepath: str = default_strategy_filepath,
        strategy_function_name: str = default_strategy_function_name,
        strategy_function: Union[FunctionType, None] = None,
        start_day: int = 1,
        end_day: int =1000,
        enable_commission: bool = True,
        graphs: List[str] = ["cum-pnl", "sharpe-heat-map", "daily-pnl"],
        prices_filepath: str = "./prices.txt",
        instruments_to_test: List[int] = range(1, 51)
    ) -> None:
        self.strategy_filepath = strategy_filepath
        self.strategy_function_name = strategy_function_name
        self.strategy_function = strategy_function
        self.start_day = start_day
        self.end_day = end_day
        self.enable_commission = enable_commission
        self.graphs = graphs
        self.prices_filepath: str = prices_filepath
        self.instruments_to_test: List[int] = instruments_to_test


# HELPER FUNCTIONS ###############################################################################
def parse_command_line_args() -> Params:
    total_args: int = len(sys.argv)
    params: Params = Params()
    if total_args > 1:
        i: int = 1
        while i < total_args:
            current_arg: str = sys.argv[i]
            if current_arg == "--path":
                if i + 1 >= total_args: raise Exception(usage_error)
                else: i += 1; params.strategy_filepath = sys.argv[i]
            elif current_arg == "--timeline":
                if i + 2 >= total_args: raise Exception(usage_error)
                else:
                    params.start_day = int(sys.argv[i + 1]); params.end_day = int(sys.argv[i + 2]); i += 2
                    if (params.start_day > params.end_day or params.start_day < 1 or params.end_day > 750):
                        raise Exception(usage_error)
            elif current_arg == "--disable-comms": params.enable_commission = False
            elif current_arg == "--function-name":
                if i + 1 >= total_args: raise Exception(usage_error)
                else: params.strategy_function_name = sys.argv[i + 1]; i += 1
            elif current_arg == "--show":
                if i + 1 >= total_args: raise Exception(usage_error)
                params.graphs = []; i += 1; current_arg = sys.argv[i]
                while current_arg not in CMD_LINE_OPTIONS:
                    if current_arg not in GRAPH_OPTIONS or len(params.graphs) == 3: raise Exception(usage_error)
                    params.graphs.append(current_arg); i += 1
                    if i < total_args: current_arg = sys.argv[i]
                    else: break
                i -= 1
            else: raise Exception(usage_error)
            i += 1
    return params

def load_get_positions_function(strategy_filepath: str, strategy_function_name: str) -> FunctionType:
    filepath: str = os.path.abspath(strategy_filepath)
    if not os.path.isfile(filepath): raise FileNotFoundError(strategy_file_not_found_message)
    module_name: str = os.path.splitext(os.path.basename(filepath))[0]
    spec: ModuleSpec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None: raise ImportError(could_not_load_spec_message)
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, strategy_function_name): raise AttributeError(strategy_function_does_not_exist_message)
    function = getattr(module, strategy_function_name)
    if not callable(function): raise TypeError(strategy_function_not_callable_message)
    return function

def generate_stats_subplot(results: BacktesterResults, subplot: Axes, enable_commission: bool) -> Axes:
    subplot.axis("off")
    win_rate_pct: float = (np.sum(results["daily_pnl"] > 0) / len(results["daily_pnl"]) * 100)
    stats_text: str = (
        f"Ran from day {results['start_day']} to {results['end_day']}\n"
        r"$\bf{Commission \ Turned \ On:}$" + f"{enable_commission}\n\n"
        r"$\bf{Backtester \ Stats}$" + "\n\n"
        f"Mean PnL: ${results['daily_pnl'].mean():.2f}\n"
        f"Std Dev: ${results['daily_pnl'].std():.2f}\n"
        f"Annualised Sharpe Ratio: "
        f"{np.sqrt(250) * results['daily_pnl'].mean() / results['daily_pnl'].std():.2f}\n"
        f"Win Rate %: {win_rate_pct:.2f}% \n"
        f"Score: {results['daily_pnl'].mean() - 0.1 * results['daily_pnl'].std():.2f}"
    )
    subplot.text(0.05, 0.95, stats_text, fontsize=14, va="top", ha="left", linespacing=1.5)
    return subplot

def generate_cumulative_pnl_subplot(results: BacktesterResults, subplot: Axes) -> Axes:
    cumulative_pnl: ndarray = np.cumsum(results["daily_pnl"])
    days: ndarray = np.arange(results["start_day"], results["start_day"] + len(cumulative_pnl))
    subplot.set_title(f"Cumulative Profit and Loss from day {results['start_day']} to {results['end_day']}", fontsize=12, fontweight="bold")
    subplot.set_xlabel("Days", fontsize=10); subplot.set_ylabel("Total PnL ($)", fontsize=10)
    subplot.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    subplot.spines["top"].set_visible(False); subplot.spines["right"].set_visible(False)
    subplot.plot(days, cumulative_pnl, linestyle="-", color=PLOT_COLORS["cum_pnl"], linewidth=2)
    return subplot

def generate_daily_pnl_subplot(results: BacktesterResults, subplot: Axes) -> Axes:
    days: ndarray = np.arange(results["start_day"], results["start_day"] + len(results["daily_pnl"]))
    subplot.set_title(f"Daily Profit and Loss (PnL) from day {results['start_day']} to {results['end_day']}", fontsize=12, fontweight="bold")
    subplot.set_xlabel("Days", fontsize=10); subplot.set_ylabel("PnL ($)", fontsize=10)
    subplot.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    subplot.spines["top"].set_visible(False); subplot.spines["right"].set_visible(False)
    subplot.plot(days, results["daily_pnl"], linestyle="-", color=PLOT_COLORS["pnl"])
    return subplot

def generate_capital_utilisation_subplot(results: BacktesterResults, subplot: Axes) -> Axes:
    daily_capital_utilisation_pct: ndarray = results["daily_capital_utilisation"] * 100
    days: ndarray = np.arange(results["start_day"], results["start_day"] + len(daily_capital_utilisation_pct))
    subplot.set_title(f"Daily capital utilisation from day {results['start_day']} to {results['end_day']}", fontsize=12, fontweight="bold")
    subplot.set_xlabel("Days", fontsize=10); subplot.set_ylabel("Capital Utilisation %", fontsize=10)
    subplot.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    subplot.spines["top"].set_visible(False); subplot.spines["right"].set_visible(False)
    subplot.set_ylim(0, 100)
    subplot.plot(days, daily_capital_utilisation_pct, linestyle="-", color=PLOT_COLORS["utilisation"])
    return subplot

def generate_sharpe_heat_map(results: BacktesterResults, subplot: Axes) -> Axes:
    returns: ndarray = results["daily_instrument_returns"]; means: ndarray = np.mean(returns, axis=1); stds: ndarray = np.std(returns, axis=1)
    sharpe_ratios: ndarray = (means / stds) * np.sqrt(250)
    sharpe_grid = sharpe_ratios.reshape(1, -1)
    im = subplot.imshow(sharpe_grid, cmap="viridis", aspect="auto")
    subplot.set_title("Annualised Sharpe-Ratio Heat Map (Higher = Better)", fontsize=12)
    subplot.set_xticks(np.arange(len(sharpe_ratios))); subplot.set_xticklabels([str(i) for i in range(len(sharpe_ratios))], fontsize=6)
    subplot.set_yticks([])
    color_bar = subplot.figure.colorbar(im, ax=subplot, orientation="vertical", pad=0.01)
    color_bar.set_label("Sharpe", fontsize=9)
    return subplot

def generate_sharpe_ratio_subplot(results: BacktesterResults, subplot: Axes) -> Axes:
    daily_pnl: ndarray = results["daily_pnl"]
    counts: ndarray = np.arange(1, len(daily_pnl) + 1)
    days: ndarray = np.arange(results["start_day"], results["start_day"] + len(daily_pnl))
    cumulative_pnl: ndarray = np.cumsum(daily_pnl); cumulative_means: ndarray = cumulative_pnl / counts
    cumulative_std_dev: ndarray = np.array([np.std(daily_pnl[: i + 1], ddof=0) if i > 0 else 1 for i in range(len(daily_pnl))])
    sharpe_ratios: ndarray = (cumulative_means / cumulative_std_dev) * np.sqrt(250)
    subplot.set_title(f"Change in Annualised Sharpe Ratio from day {results['start_day']} to {results['end_day']}", fontsize=12, fontweight="bold")
    subplot.set_xlabel("Days", fontsize=10); subplot.set_ylabel("Annualised Sharpe Ratio", fontsize=10)
    subplot.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    subplot.spines["top"].set_visible("False"); subplot.spines["right"].set_visible(False)
    subplot.plot(days, sharpe_ratios, linestyle="-", color=PLOT_COLORS["sharpe_change"])
    return subplot

def get_subplot(graph_type: str, results: BacktesterResults, subplot: Axes) -> Axes:
    if graph_type == "daily-pnl": return generate_daily_pnl_subplot(results, subplot)
    elif graph_type == "cum-pnl": return generate_cumulative_pnl_subplot(results, subplot)
    elif graph_type == "capital-util": return generate_capital_utilisation_subplot(results, subplot)
    elif graph_type == "sharpe-heat-map": return generate_sharpe_heat_map(results, subplot)
    elif graph_type == "cum-sharpe": return generate_sharpe_ratio_subplot(results, subplot)
    return subplot


def get_ema(instrument_price_history: ndarray, lookback: int) -> ndarray:
    price_series: Series = pd.Series(instrument_price_history)
    return price_series.ewm(span=lookback, adjust=False).mean()


# BACKTESTER CLASS ################################################################################
class Backtester:
    def __init__(self, params: Params) -> None:
        self.enable_commission: bool = params.enable_commission
        # MODIFIED: State is now tracked with an array for entry prices.
        self.entry_prices = np.zeros(NUMBER_OF_INSTRUMENTS)
        self.getMyPosition: Union[FunctionType, None]
        if params.strategy_function is not None:
            self.getMyPosition = params.strategy_function
        else:
            self.getMyPosition = load_get_positions_function(params.strategy_filepath, params.strategy_function_name)
        self.raw_prices_df: DataFrame = pd.read_csv(params.prices_filepath, sep=r"\s+", header=None)
        self.price_history: ndarray = self.raw_prices_df.to_numpy().T

    def run(
        self,
        start_day: int,
        end_day: int,
        config: Union[Dict[int, Dict[str, Dict[str, float]]], None] = None,
        instruments_to_test: Union[List[int], None] = None
    ) -> BacktesterResults:
        # Initialize variables
        cash: float = 0
        curPos: ndarray = np.zeros(NUMBER_OF_INSTRUMENTS)
        totDVolume: float = 0
        value: float = 0
        todayPLL: List[float] = []
        
        # Additional tracking
        daily_capital_utilisation_list: List[float] = []
        instrument_returns = {instrument: [] for instrument in range(50)}
        trades = {instrument: [] for instrument in range(50)}
        
        for t in range(start_day, end_day + 1):
            prcHistSoFar: ndarray = self.price_history[:, :t]
            curPrices: ndarray = prcHistSoFar[:, -1]
            
            if t < end_day:
                
                newPosOrig: ndarray = self.getMyPosition(prcHistSoFar, curPos, self.entry_prices)
                
                posLimits: ndarray = np.array([int(x) for x in INSTRUMENT_POSITION_LIMIT / curPrices])
                newPos: ndarray = np.clip(newPosOrig, -posLimits, posLimits)
                
                deltaPos: ndarray = newPos - curPos
                
                
                for i in range(NUMBER_OF_INSTRUMENTS):
                    if deltaPos[i] != 0 and curPos[i] == 0:  # A new position was opened
                        self.entry_prices[i] = curPrices[i]
                    elif deltaPos[i] != 0 and newPos[i] == 0:  # A position was closed
                        self.entry_prices[i] = 0
                
                dvolumes: ndarray = curPrices * np.abs(deltaPos)
                dvolume: float = np.sum(dvolumes)
                totDVolume += dvolume
                
                comm: float = dvolume * COMMISSION_RATE if self.enable_commission else 0
                
                cash -= curPrices.dot(deltaPos) + comm
                
                for i in range(50):
                    if deltaPos[i] != 0:
                        trades[i].append({
                            "price_entry": curPrices[i],
                            "order_type": "BUY" if deltaPos[i] > 0 else "SELL",
                            "day": t
                        })
                
                capital_utilisation: float = dvolume / (INSTRUMENT_POSITION_LIMIT * NUMBER_OF_INSTRUMENTS)
                daily_capital_utilisation_list.append(capital_utilisation)
            else:
                newPos = np.array(curPos)
                daily_capital_utilisation_list.append(0)
            
            curPos = np.array(newPos)
            
            posValue: float = curPos.dot(curPrices)
            # P&L is the change in portfolio value from yesterday to today
            todayPL: float = cash + posValue - value
            # Update value for the next day's calculation
            value = cash + posValue
            
            if t > start_day:
                todayPLL.append(todayPL)
            
            for i in range(50):
                if t > start_day and curPos[i] != 0:
                    yesterday_price = self.price_history[i, t - 2]
                    instrument_return = curPos[i] * (curPrices[i] - yesterday_price)
                    instrument_returns[i].append(instrument_return)
                else:
                    instrument_returns[i].append(0)
        
        pll: ndarray = np.array(todayPLL)
        
        backtester_results: BacktesterResults = {}
        backtester_results["daily_pnl"] = pll
        backtester_results["daily_capital_utilisation"] = np.array(daily_capital_utilisation_list)
        backtester_results["daily_instrument_returns"] = np.array([instrument_returns[i] for i in range(50)])
        backtester_results["trades"] = trades
        backtester_results["start_day"] = start_day
        backtester_results["end_day"] = end_day
        
        return backtester_results

    def show_dashboard(self, backtester_results: BacktesterResults, graphs: List[str]) -> None:
        fig, axs = plt.subplots(2, 2, figsize=(18, 8))
        axs[0][0] = generate_stats_subplot(backtester_results, axs[0][0], self.enable_commission)
        axs[0][1] = get_subplot(graphs[0], backtester_results, axs[0][1])
        if len(graphs) > 1: axs[1][0] = get_subplot(graphs[1], backtester_results, axs[1][0])
        else: axs[1][0].axis("off")
        if len(graphs) > 2: axs[1][1] = get_subplot(graphs[2], backtester_results, axs[1][1])
        else: axs[1][1].axis("off")
        plt.tight_layout(); plt.subplots_adjust(top=0.88)
        plt.suptitle("Backtest Performance Summary", fontsize=16, fontweight="bold")
        plt.show()

    def show_price_entries(self, backtester_results: BacktesterResults) -> None:
        pass

# MAIN EXECUTION #################################################################################
def main_func() -> None:
    params: Params = parse_command_line_args()
    backtester: Backtester = Backtester(params)
    backtester_results: BacktesterResults = backtester.run(
        params.start_day,
        params.end_day
    )
    backtester.show_dashboard(backtester_results,
        params.graphs)


if __name__ == "__main__":
    main_func()