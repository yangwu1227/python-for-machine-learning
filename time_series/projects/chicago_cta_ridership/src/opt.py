import pandas as pd
import pulp


def optimize_vehicle_allocation(
    forecast: pd.DataFrame,
    cost_bus_40: int,
    cost_bus_60: int,
    cost_rail: int,
    capacity_bus_40: int,
    capacity_bus_60: int,
    capacity_rail: int,
    trips_per_bus: int,
    trips_per_rail: int,
    percentage_40_foot: float,
) -> pd.DataFrame:
    """
    Optimize the allocation of buses and railcars based on forecasted demand.

    Parameters
    ----------
    forecast : pd.DataFrame
        Forecast data with 'bus' and 'rail_boardings' columns.
    cost_bus_40 : int
        Operating cost per day for a 40-foot bus.
    cost_bus_60 : int
        Operating cost per day for a 60-foot bus.
    cost_rail : int
        Operating cost per day for a railcar.
    capacity_bus_40 : int
        Capacity of a 40-foot bus.
    capacity_bus_60 : int
        Capacity of a 60-foot bus.
    capacity_rail : int
        Capacity of a railcar.
    trips_per_bus : int
        Number of trips a bus can make in a day.
    trips_per_rail : int
        Number of trips a railcar can make in a day.
    percentage_40_foot : float
        Percentage of 40-foot buses in the final solution.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the optimal number of buses and railcars for each day.
    """
    days = forecast.index

    # Problem formulation
    problem = pulp.LpProblem("Vehicle_Allocation", pulp.LpMinimize)

    # Decision variables
    buses_40 = pulp.LpVariable.dicts("Buses_40", days, lowBound=0, cat="Integer")
    buses_60 = pulp.LpVariable.dicts("Buses_60", days, lowBound=0, cat="Integer")
    rails = pulp.LpVariable.dicts("Rails", days, lowBound=0, cat="Integer")

    # Objective function
    problem += pulp.lpSum(
        [
            cost_bus_40 * buses_40[day]
            + cost_bus_60 * buses_60[day]
            + cost_rail * rails[day]
            for day in days
        ]
    )

    # Constraints
    for day in days:
        forecasted_bus = forecast.loc[day, "bus"]
        problem += (
            buses_40[day] * capacity_bus_40 + buses_60[day] * capacity_bus_60
        ) * trips_per_bus >= forecasted_bus

        forecasted_rail = forecast.loc[day, "rail_boardings"]
        problem += rails[day] * capacity_rail * trips_per_rail >= forecasted_rail

        # Additional constraint for 40-foot buses (modify as needed)
        problem += buses_40[day] >= percentage_40_foot * (buses_40[day] + buses_60[day])

    # Solve the problem
    problem.solve()

    # Extracting the solution
    optimal_solution = {
        "Date": days,
        "Optimal Number of 40-foot Buses": [buses_40[day].varValue for day in days],
        "Optimal Number of 60-foot Buses": [buses_60[day].varValue for day in days],
        "Optimal Number of Railcars": [rails[day].varValue for day in days],
    }

    return pd.DataFrame(optimal_solution)
