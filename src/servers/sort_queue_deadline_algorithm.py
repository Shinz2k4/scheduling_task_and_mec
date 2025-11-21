import asyncio


def sort_queue_deadline(queue: asyncio.Queue):
    """
    thuật toán tối ưu lập lịch 
    """
    print("Sorting queue ...")
    if queue.empty():
        return []
    
    tasks = list(queue._queue)
    N = len(tasks)
    queue_sorted = [0] * N
    
    tasks = sorted(tasks, key=lambda t: t.deadline) # tap b
    tasks_sorted_by_required_cpu = sorted(tasks, key=lambda t: t.require_cpu) # tap c

    for i, task in enumerate(tasks_sorted_by_required_cpu):
        i_star = 0
        for idx, t in enumerate(tasks):
            if t.id == task.id:
                i_star = idx
                break
          
        queue_sorted[i_star] = task
        outage = False
    
        for j in range(i_star, N):
            required_cpu = 0
            deadline = 0
            required_cpu = getattr(task, "require_cpu")
            deadline = getattr(task, "deadline")

            total = sum(required_cpu * queue_sorted[k] / (3.1*10**9) for k in range(j + 1))
            
            if queue_sorted[j] != 0 and total > deadline * queue_sorted[j]:
                outage = True
                break
        if outage:
            queue_sorted[i_star] = 0
            print(f"Task {task.id} bị quá deadline.")
    queue_sorted = [task for task in queue_sorted if task != 0]
    return queue_sorted