def schedule_pipeline(tasks, resource_budget):
    """
    Schedule ETL tasks respecting dependencies and resource limits.
    """
    task_map = {task["name"]: task for task in tasks}
    completed = set()
    started = set()
    running = []  # list of (end_time, task_name, resources)
    schedule = []
    time = 0

    while len(completed) < len(tasks):
        # Complete all tasks finishing now
        finished_now = [item for item in running if item[0] == time]
        if finished_now:
            for _, name, _ in finished_now:
                completed.add(name)
            running = [item for item in running if item[0] != time]

        # Find ready tasks
        ready = []
        for task in tasks:
            name = task["name"]
            if name in started:
                continue
            if all(dep in completed for dep in task["depends_on"]):
                ready.append(task)

        ready.sort(key=lambda t: t["name"])

        # Current resource usage
        used_resources = sum(res for _, _, res in running)

        # Greedily start tasks
        for task in ready:
            need = task["resources"]
            if used_resources + need <= resource_budget:
                name = task["name"]
                start_time = time
                end_time = time + task["duration"]

                started.add(name)
                running.append((end_time, name, need))
                schedule.append((name, start_time))
                used_resources += need

        # If all done after scheduling/completing, stop
        if len(completed) == len(tasks):
            break

        # Advance to next completion event
        if running:
            time = min(end_time for end_time, _, _ in running)

    return sorted(schedule, key=lambda x: (x[1], x[0]))