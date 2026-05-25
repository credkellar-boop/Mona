import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription

class LiveRoomManager:
    """
    Manages WebRTC SFU sessions for 1-on-1 chats, group video chats (max 50), 
    and massive live streaming broadcast topologies.
    """
    def __init__(self):
        # Maps roomId -> set of connected participant peer connections
        self.rooms = {}
        # Maps roomId -> list of remote media tracks to forward
        self.room_tracks = {}
        self.MAX_PARTICIPANTS = 50

    async def join_room(self, room_id: str, client_id: str, offer_sdp: str, is_publisher: bool = True):
        if room_id not in self.rooms:
            self.rooms[room_id] = {}
            self.room_tracks[room_id] = []

        if len(self.rooms[room_id]) >= self.MAX_PARTICIPANTS:
            raise ValueError(f"Room {room_id} has reached its maximum capacity of {self.MAX_PARTICIPANTS} participants.")

        pc = RTCPeerConnection()
        self.rooms[room_id][client_id] = pc

        @pc.on("track")
        def on_track(track):
            print(f"📡 Received new video/audio track in Room {room_id} from {client_id}")
            if is_publisher:
                self.room_tracks[room_id].append(track)
                # SFU Logic: Forward this newly added track to all other connected subscribers in the room
                self._broadcast_track_to_subscribers(room_id, track, skip_client=client_id)

        # Handle browser signaling negotiation
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        await pc.setRemoteDescription(offer)
        
        # Attach existing room tracks to this new participant so they can see everyone else
        for existing_track in self.room_tracks[room_id]:
            pc.addTrack(existing_track)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return pc.localDescription.sdp

    def _broadcast_track_to_subscribers(self, room_id: str, track, skip_client: str):
        for peer_id, pc in self.rooms[room_id].items():
            if peer_id != skip_client:
                try:
                    pc.addTrack(track)
                except Exception as e:
                    print(f"Error forwarding track to peer {peer_id}: {e}")

    def leave_room(self, room_id: str, client_id: str):
        if room_id in self.rooms and client_id in self.rooms[room_id]:
            pc = self.rooms[room_id].pop(client_id)
            asyncio.ensure_future(pc.close())
            print(f"🚪 Participant {client_id} left Room {room_id}")
