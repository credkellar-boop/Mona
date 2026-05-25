# live_server.py
import os
import uuid
from aiohttp import web
from mona.utils.live_sfu import LiveRoomManager

room_manager = LiveRoomManager()

async def handle_join(request):
    """
    Handles WebRTC initialization negotiation requests for group video chats and live streams.
    """
    try:
        data = await request.json()
        room_id = data.get("room_id", "default-lounge")
        offer_sdp = data.get("sdp")
        is_publisher = data.get("is_publisher", True)
        client_id = str(uuid.uuid4())[:8]

        # Process standard 1-on-1 or multi-party room entry via the SFU engine
        answer_sdp = await room_manager.join_room(
            room_id=room_id, 
            client_id=client_id, 
            offer_sdp=offer_sdp, 
            is_publisher=is_publisher
        )

        return web.json_response({
            "status": "success", 
            "client_id": client_id, 
            "sdp": answer_sdp
        })

    except ValueError as err:
        return web.json_response({"status": "rejected", "reason": str(err)}, status=403)
    except Exception as e:
        return web.json_response({"status": "error", "reason": str(e)}, status=500)

async def handle_leave(request):
    data = await request.json()
    room_manager.leave_room(data.get("room_id"), data.get("client_id"))
    return web.json_response({"status": "disconnected"})

# Setup web endpoints for your layout documentation
app = web.Application()
app.router.add_post("/api/live/join", handle_join)
app.router.add_post("/api/live/leave", handle_leave)

if __name__ == "__main__":
    print("🌐 Mona Live Streaming & Group Video Server starting on port 8080...")
    web.run_app(app, port=8080)
