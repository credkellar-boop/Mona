import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

class LiveRoomManager:
    """
    Manages WebRTC SFU sessions for 1-on-1 chats, group video chats (max 50), 
    and massive live streaming broadcast topologies.
    """
    def __init__(self):
        # Maps roomId -> dict of {client_id: RTCPeerConnection}
        self.rooms = {}
        # Maps roomId -> list of relayed MediaStreamTracks to proxy out
        self.room_tracks = {}
        # Central media proxy to duplicate track streams without decoding overhead
        self.relay = MediaRelay()
        self.MAX_PARTICIPANTS = 50

    async def join_room(self, room_id: str, client_id: str, offer_sdp: str, is_publisher: bool = True):
        if room_id not in self.rooms:
            self.rooms[room_id] = {}
            self.room_tracks[room_id] = []

        if len(self.rooms[room_id]) >= self.MAX_PARTICIPANTS:
            raise ValueError(f"Room {room_id} has reached maximum capacity of {self.MAX_PARTICIPANTS} participants.")

        pc = RTCPeerConnection()
        self.rooms[room_id][client_id] = pc

        @pc.on("track")
        def on_track(track):
            print(f"📡 Received incoming {track.kind} track in Room {room_id} from {client_id}")
            if is_publisher:
                # Wrap the track inside the relay proxy before sharing it
                relayed_track = self.relay.subscribe(track)
                self.room_tracks[room_id].append(relayed_track)
                
                # SFU Logic: Forward this newly relayed track to all other connected subscribers
                self._broadcast_track_to_subscribers(room_id, relayed_track, skip_client=client_id)

        # Handle browser signaling negotiation handshake
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        await pc.setRemoteDescription(offer)
        
        # Attach all existing room tracks to this new participant on entry
        for existing_track in self.room_tracks[room_id]:
            pc.addTrack(existing_track)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return pc.localDescription.sdp

    def _broadcast_track_to_subscribers(self, room_id: str, relayed_track, skip_client: str):
        for peer_id, pc in self.rooms[room_id].items():
            if peer_id != skip_client:
                try:
                    pc.addTrack(relayed_track)
                    print(f"   -> Forwarded track to subscriber: {peer_id}")
                except Exception as e:
                    print(f"⚠️ Error forwarding track to peer {peer_id}: {e}")

    def leave_room(self, room_id: str, client_id: str):
        if room_id in self.rooms and client_id in self.rooms[room_id]:
            pc = self.rooms[room_id].pop(client_id)
            asyncio.ensure_future(pc.close())
            print(f"🚪 Participant {client_id} left Room {room_id}")
