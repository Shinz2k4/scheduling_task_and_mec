import time

def fast_detect_outage(queues,id, cpu_required, deadline):
    """
    Kiểm tra xem task có thể lập lịch được hay không
    """
    f_server = 3.1*10**9  
    delay = time.time()
    start_time = time.time()
    server_outage = 0
    
    new_queues = sorted(queues + [{'id': id, 'cpu_required': cpu_required, 'deadline': deadline}],
                key=lambda t: t['deadline'])
    if len(new_queues) == 0:
        new_queues = []
        
    for task in new_queues:
        delay += cpu_required / f_server
        if delay > (deadline + start_time):
            print(f"Task {id} không thể lập lịch tại server này (vượt deadline {deadline})")
            break
        else:
            server_outage = 1
    if server_outage == 1:  
        print(f"Task {id} lập lịch được (delay cuối = {delay:.2f})")
        return 0,delay
    return 1,delay
