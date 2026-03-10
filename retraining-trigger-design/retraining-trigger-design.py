def retraining_policy(daily_stats, config):
    """
    Decide which days to trigger model retraining.
    """
    drift_threshold = config["drift_threshold"]
    performance_threshold = config["performance_threshold"]
    max_staleness = config["max_staleness"]
    cooldown = config["cooldown"]
    retrain_cost = config["retrain_cost"]
    budget = config["budget"]

    triggered_days = []
    days_since_retrain = 0
    last_retrain_day = None

    for stats in daily_stats:
        day = stats["day"]
        days_since_retrain += 1

        drift_trigger = stats["drift_score"] > drift_threshold
        performance_trigger = stats["performance"] < performance_threshold
        staleness_trigger = days_since_retrain >= max_staleness

        should_retrain = drift_trigger or performance_trigger or staleness_trigger

        cooldown_ok = (
            last_retrain_day is None or (day - last_retrain_day) >= cooldown
        )
        budget_ok = budget >= retrain_cost

        if should_retrain and cooldown_ok and budget_ok:
            triggered_days.append(day)
            budget -= retrain_cost
            days_since_retrain = 0
            last_retrain_day = day

    return triggered_days