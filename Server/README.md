# Server - For developer stage
You can use docker container to run this websocket server, use following step:
1. run container, and set volume to now path (you can use pwd command to get now path).
websocket port is set to 5566 by default value(also can set env to overwrite), so use this command:

```bash
$ docker run -it --name ws-server -d -v  $(pwd):/workspace -p 5566:5566 python:3.7
```

2. exeute into container, and run server.py

```bash
$ docker exec -it ws-server bash

# in container
$ cd /workspace/Server
$ pip install -r requirements.txt
$ python server.py
```

3. use [websocket test site](https://www.websocket.org/echo.html) to test connection.

actionType List
- register: For FE on connect use.
- unregister: For FE on disconnect use.
- broadcast: For broadcast(send msg to all registered users) use.
- relay: For call music generator process and return music use.

* Location: `ws://127.0.0.1:5566`
* Message: `{"action": "<actionType>", "payload": "<payload>"}`


