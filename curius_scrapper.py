import requests, time, json
from collections import deque

USER_EP = "https://curius.app/api/users/vishal-waghmode"       # from DevTools
FOLLOW_EP = "https://curius.app/api/...{user_id}"  # from DevTools

seen, queue, out = set(), deque(["seed-profile-slug"]), []
while queue:
    slug = queue.popleft()
    if slug in seen: continue
    seen.add(slug)
    user = requests.get(USER_EP.format(slug=slug), timeout=15).json()
    out.append(user)
    for f in requests.get(FOLLOW_EP.format(user_id=user["id"]), timeout=15).json():
        queue.append(f["userLink"])   # adjust key name to the real JSON
    time.sleep(1)                     # 1 request per second
json.dump(out, open("curius_users.json", "w"))