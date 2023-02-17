unique_keys = []

f = open("storage_keys.txt", "r")
for k in f.readlines():
    k_arr = k.split(",")
    #print(k_arr)
    k_action = []
    for i in range(len(k_arr)):
        key = bool(int(k_arr[i][1]))
        k_action.append(key)
    #print(k_action)
    unique_keys.append(k_action)
f.close()
