"""
WebSocket handler - Direct replacement for Gradio real-time logs
"""
import asyncio
import json
import logging
import queue
import threading
from datetime import datetime
from typing import Set

from src.utils.log_utils import reformat
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from api.models.schemas import WebSocketMessage, LogMessage

router = APIRouter()

# Global connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.log_queue = queue.Queue()
        self.logger_setup = False
        self.log_handler = None  # Track handler to prevent duplicates

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Setup logging ONLY on first connection
        if not self.logger_setup:
            self.setup_logging()
            self.logger_setup = True

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logging.error(f"Error sending message to WebSocket: {e}")
                self.disconnect(websocket)

    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSockets"""
        disconnected = set()
        for connection in self.active_connections:
            if connection.client_state == WebSocketState.CONNECTED:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logging.error(f"Error broadcasting to WebSocket: {e}")
                    disconnected.add(connection)
            else:
                disconnected.add(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

    def setup_logging(self):
        """Setup logging to capture logs for WebSocket - PREVENT DUPLICATES"""
        if self.log_handler:
            # Handler already exists, don't create another
            return
            
        class WebSocketLogHandler(logging.Handler):
            def __init__(self, connection_manager):
                super().__init__()
                self.connection_manager = connection_manager

            def emit(self, record):
                try:
                    # Skip our own WebSocket log messages to prevent recursion
                    if record.name.startswith('uvicorn') or 'WebSocket' in record.getMessage():
                        return
                        
                    log_entry = self.format(record)
                    formatted_message = self.reformat_message(log_entry)
                    
                    # Create WebSocket message
                    ws_message = WebSocketMessage(
                        type="log",
                        data={
                            "level": record.levelname,
                            "message": record.getMessage(),
                            "formatted_message": formatted_message,
                            "timestamp": datetime.now().isoformat()
                        },
                        timestamp=datetime.now().isoformat()
                    )
                    
                    # Add to queue for broadcasting
                    self.connection_manager.log_queue.put(ws_message.json())
                    
                except Exception as e:
                    # Use print instead of logging to prevent recursion
                    print(f"Error in WebSocket log handler: {e}")

            def reformat_message(self, message: str) -> str:
                """Reformat log message for display"""
                try:
                    return reformat(message)
                except ImportError:
                    # Fallback formatting if log_utils is not available
                    formatted = message.replace('[INFO]', '<span style="color: #4CAF50;">[INFO]</span>')
                    formatted = formatted.replace('[ERROR]', '<span style="color: #F44336;">[ERROR]</span>')
                    formatted = formatted.replace('[WARNING]', '<span style="color: #FF9800;">[WARNING]</span>')
                    return formatted
                except Exception:
                    return message

        # Create and store the handler
        self.log_handler = WebSocketLogHandler(self)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %z",
        )
        self.log_handler.setFormatter(formatter)
        
        # Add handler to specific logger instead of root to prevent duplicates
        # Get the 'Agents' logger (based on your logs showing [Agents])
        agents_logger = logging.getLogger("Agents")
        agents_logger.addHandler(self.log_handler)
        agents_logger.setLevel(logging.INFO)
        
        # Also add to root logger but with higher level to avoid too many logs
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        
        # Start background task to process log queue
        asyncio.create_task(self.process_log_queue())

    async def process_log_queue(self):
        """Process log queue and broadcast to all connections"""
        while True:
            try:
                if not self.log_queue.empty():
                    log_message = self.log_queue.get_nowait()
                    await self.broadcast(log_message)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error processing log queue: {e}")  # Use print to avoid logging recursion
                await asyncio.sleep(1)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/logs")
async def websocket_logs_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time logs - Direct replacement for Gradio HTML logs
    """
    await manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        welcome_message = WebSocketMessage(
            type="status",
            data={"message": "Connected to DealMind AI logs", "connected": True},
            timestamp=datetime.now().isoformat()
        )
        await manager.send_personal_message(welcome_message.json(), websocket)
        
        # Keep connection alive and handle any incoming messages
        while True:
            try:
                # Wait for messages from client (ping, etc.)
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle different message types
                if message_data.get("type") == "ping":
                    pong_response = WebSocketMessage(
                        type="pong",
                        data={"timestamp": datetime.now().isoformat()},
                        timestamp=datetime.now().isoformat()
                    )
                    await manager.send_personal_message(pong_response.json(), websocket)
                    
                elif message_data.get("type") == "request_history":
                    # Send recent log history if needed
                    history_response = WebSocketMessage(
                        type="history",
                        data={"logs": [], "message": "Log history not implemented yet"},
                        timestamp=datetime.now().isoformat()
                    )
                    await manager.send_personal_message(history_response.json(), websocket)
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                error_response = WebSocketMessage(
                    type="error",
                    data={"error": "Invalid JSON message"},
                    timestamp=datetime.now().isoformat()
                )
                await manager.send_personal_message(error_response.json(), websocket)
            except Exception as e:
                print(f"Error in WebSocket message handling: {e}")  # Use print to avoid logging recursion
                error_response = WebSocketMessage(
                    type="error",
                    data={"error": f"Message handling error: {str(e)}"},
                    timestamp=datetime.now().isoformat()
                )
                await manager.send_personal_message(error_response.json(), websocket)
                
    except WebSocketDisconnect:
        pass  # Don't log disconnect to avoid noise
    except Exception as e:
        print(f"WebSocket error: {e}")  # Use print to avoid logging recursion
    finally:
        manager.disconnect(websocket)


@router.websocket("/status")
async def websocket_status_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time status updates
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Send periodic status updates
            status_message = WebSocketMessage(
                type="status",
                data={
                    "active_connections": len(manager.active_connections),
                    "timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now().isoformat()
            )
            await manager.send_personal_message(status_message.json(), websocket)
            
            # Wait 30 seconds before next status update
            await asyncio.sleep(30)
            
    except WebSocketDisconnect:
        pass  # Don't log disconnect to avoid noise
    except Exception as e:
        print(f"Status WebSocket error: {e}")  # Use print to avoid logging recursion
    finally:
        manager.disconnect(websocket)


# Helper function to send custom messages through WebSocket
async def broadcast_custom_message(message_type: str, data: dict):
    """
    Broadcast custom message to all connected clients
    """
    custom_message = WebSocketMessage(
        type=message_type,
        data=data,
        timestamp=datetime.now().isoformat()
    )
    await manager.broadcast(custom_message.json())