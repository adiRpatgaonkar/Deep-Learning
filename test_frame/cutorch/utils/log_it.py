
def time_log(t):
    if t > 3600:
        return str(round(t/3600)) + " hours"
    elif t > 60:
        return str(round(t/60.00, 2)) + " minutes"
    else:
        return str(round(t, 2)) + " seconds"