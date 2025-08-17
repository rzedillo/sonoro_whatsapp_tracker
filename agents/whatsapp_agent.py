"""
WhatsApp Agent for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
Ported from whatsapp-web.js to Python Selenium
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import undetected_chromedriver as uc

from core.base_agent import BaseAgent
from core.redis_client import cache_manager
from database.connection import get_db_context
from database.models import WhatsAppSession, Conversation
from agents.whatsapp_command_handler import WhatsAppCommandHandler


class WhatsAppAgent(BaseAgent):
    """
    WhatsApp Web integration agent using Selenium
    
    Features:
    - QR code authentication
    - Message monitoring from specified groups
    - Session persistence
    - Automatic reconnection
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # WhatsApp specific configuration
        self.session_path = config.get("session_path", "./data/whatsapp_session")
        self.qr_timeout = config.get("qr_timeout", 60)
        self.headless = config.get("headless", True)
        self.monitored_groups = config.get("monitored_groups", [])
        
        # Browser and session management
        self.driver: Optional[webdriver.Chrome] = None
        self.session_id = f"whatsapp_session_{int(time.time())}"
        self.is_authenticated = False
        self.phone_number = None
        
        # Message monitoring
        self.last_message_time = datetime.utcnow()
        self.message_history = {}
        self.monitoring_task = None
        
        # Command handling (from v1)
        self.command_handler = None  # Will be initialized after orchestrator is available
        
        # Message sending capabilities
        self.send_queue = []
        self.rate_limit_delay = config.get("rate_limit_delay", 1000)  # ms between messages
        
        # Ensure session directory exists
        Path(self.session_path).mkdir(parents=True, exist_ok=True)
    
    async def _initialize_agent(self):
        """Initialize WhatsApp agent"""
        self.logger.info("Initializing WhatsApp agent")
        
        try:
            # Set up Chrome driver
            await self._setup_chrome_driver()
            
            # Try to restore existing session
            restored = await self._restore_session()
            
            if not restored:
                # Start fresh authentication
                await self._start_authentication()
            
            # Start message monitoring
            await self._start_monitoring()
            
            self.logger.info("WhatsApp agent initialized successfully")
            
        except Exception as e:
            self.logger.error("WhatsApp agent initialization failed", error=str(e))
            raise
    
    async def _cleanup_agent(self):
        """Cleanup WhatsApp agent"""
        self.logger.info("Cleaning up WhatsApp agent")
        
        try:
            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Save session data
            await self._save_session()
            
            # Close browser
            if self.driver:
                self.driver.quit()
                self.driver = None
            
            self.logger.info("WhatsApp agent cleanup completed")
            
        except Exception as e:
            self.logger.error("WhatsApp agent cleanup error", error=str(e))
    
    def set_orchestrator(self, orchestrator):
        """Set orchestrator reference for command handling"""
        self.command_handler = WhatsAppCommandHandler(orchestrator)
        self.logger.info("Command handler initialized")
    
    async def _process_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process WhatsApp-related commands"""
        command = data.get("command", "")
        
        try:
            if command == "get_qr_code":
                return await self._get_qr_code()
            elif command == "get_status":
                return await self._get_whatsapp_status()
            elif command == "send_message":
                return await self._send_message(data)
            elif command == "get_chats":
                return await self._get_chats()
            elif command == "reconnect":
                return await self._reconnect()
            elif command == "set_orchestrator":
                self.set_orchestrator(data.get("orchestrator"))
                return {"success": True, "message": "Orchestrator set"}
            else:
                return {
                    "success": False,
                    "error": f"Unknown command: {command}"
                }
                
        except Exception as e:
            self.logger.error("WhatsApp command processing failed", command=command, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "command": command
            }
    
    async def _health_check(self) -> Dict[str, Any]:
        """WhatsApp agent health check"""
        try:
            return {
                "authenticated": self.is_authenticated,
                "driver_active": self.driver is not None,
                "session_id": self.session_id,
                "phone_number": self.phone_number,
                "monitored_groups": len(self.monitored_groups),
                "last_message_time": self.last_message_time.isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _setup_chrome_driver(self):
        """Set up Chrome driver with WhatsApp Web optimizations"""
        try:
            self.logger.info("Setting up Chrome driver")
            
            chrome_options = Options()
            
            # Basic options
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            
            # WhatsApp Web specific options
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            
            # User data directory for session persistence
            user_data_dir = os.path.join(self.session_path, "chrome_profile")
            chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
            
            # Headless mode
            if self.headless:
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--window-size=1920,1080")
            
            # Create driver
            self.driver = uc.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
            
            self.logger.info("Chrome driver setup completed")
            
        except Exception as e:
            self.logger.error("Chrome driver setup failed", error=str(e))
            raise
    
    async def _restore_session(self) -> bool:
        """Try to restore existing WhatsApp session"""
        try:
            self.logger.info("Attempting to restore WhatsApp session")
            
            # Load WhatsApp Web
            self.driver.get("https://web.whatsapp.com")
            
            # Wait a moment for page to load
            await asyncio.sleep(3)
            
            # Check if already authenticated
            try:
                # Look for chat list or main interface elements
                WebDriverWait(self.driver, 10).until(
                    EC.any_of(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='chat-list']")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='default-user']")),
                    )
                )
                
                self.is_authenticated = True
                self.logger.info("WhatsApp session restored successfully")
                
                # Try to get phone number
                await self._get_phone_number()
                
                return True
                
            except TimeoutException:
                self.logger.info("No existing session found")
                return False
                
        except Exception as e:
            self.logger.error("Session restoration failed", error=str(e))
            return False
    
    async def _start_authentication(self):
        """Start fresh WhatsApp authentication with QR code"""
        try:
            self.logger.info("Starting WhatsApp authentication")
            
            # Load WhatsApp Web if not already loaded
            if self.driver.current_url != "https://web.whatsapp.com":
                self.driver.get("https://web.whatsapp.com")
            
            # Wait for QR code to appear
            qr_code_element = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "canvas[aria-label='Scan me!']"))
            )
            
            # Get QR code data
            qr_code_data = await self._extract_qr_code()
            
            if qr_code_data:
                # Save QR code to session
                await self._save_qr_code(qr_code_data)
                
                self.logger.info("QR code generated, waiting for authentication")
                
                # Wait for authentication
                await self._wait_for_authentication()
            
        except Exception as e:
            self.logger.error("Authentication failed", error=str(e))
            raise
    
    async def _extract_qr_code(self) -> Optional[str]:
        """Extract QR code as base64 image"""
        try:
            # Take screenshot of QR code element
            qr_element = self.driver.find_element(By.CSS_SELECTOR, "canvas[aria-label='Scan me!']")
            qr_screenshot = qr_element.screenshot_as_base64
            
            return qr_screenshot
            
        except Exception as e:
            self.logger.error("QR code extraction failed", error=str(e))
            return None
    
    async def _wait_for_authentication(self):
        """Wait for user to scan QR code and authenticate"""
        try:
            self.logger.info("Waiting for QR code scan")
            
            # Wait for main interface to appear
            WebDriverWait(self.driver, self.qr_timeout).until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='chat-list']")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='default-user']")),
                )
            )
            
            self.is_authenticated = True
            self.logger.info("WhatsApp authentication successful")
            
            # Get phone number
            await self._get_phone_number()
            
            # Save session
            await self._save_session()
            
        except TimeoutException:
            self.logger.error("Authentication timeout - QR code not scanned")
            raise Exception("Authentication timeout")
    
    async def _get_phone_number(self):
        """Extract phone number from WhatsApp interface"""
        try:
            # Try to find phone number in profile or interface
            # This might need adjustment based on WhatsApp Web interface changes
            phone_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='default-user']")
            
            if phone_elements:
                # Phone number extraction logic here
                self.phone_number = "extracted_number"  # Placeholder
                
        except Exception as e:
            self.logger.warning("Could not extract phone number", error=str(e))
    
    async def _start_monitoring(self):
        """Start monitoring WhatsApp messages"""
        if not self.is_authenticated:
            self.logger.warning("Cannot start monitoring - not authenticated")
            return
        
        self.logger.info("Starting WhatsApp message monitoring")
        
        # Create monitoring task
        self.monitoring_task = asyncio.create_task(self._monitor_messages())
    
    async def _monitor_messages(self):
        """Monitor WhatsApp messages in background"""
        try:
            while self.is_authenticated:
                # Check for new messages
                await self._check_new_messages()
                
                # Wait before next check
                await asyncio.sleep(2)
                
        except asyncio.CancelledError:
            self.logger.info("Message monitoring cancelled")
        except Exception as e:
            self.logger.error("Message monitoring error", error=str(e))
    
    async def _check_new_messages(self):
        """Check for new messages in monitored groups"""
        try:
            # Get all chat elements
            chat_elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                "[data-testid='chat-list'] > div > div"
            )
            
            for chat_element in chat_elements:
                try:
                    # Get chat name
                    chat_name = self._get_chat_name(chat_element)
                    
                    # Check if this is a monitored group
                    if chat_name in self.monitored_groups:
                        # Check for unread messages
                        await self._process_chat_messages(chat_element, chat_name)
                        
                except Exception as e:
                    self.logger.error("Error processing chat", error=str(e))
                    continue
                    
        except Exception as e:
            self.logger.error("Error checking messages", error=str(e))
    
    def _get_chat_name(self, chat_element) -> str:
        """Extract chat name from chat element"""
        try:
            name_element = chat_element.find_element(
                By.CSS_SELECTOR,
                "[data-testid='conversation-info-header']"
            )
            return name_element.text
        except NoSuchElementException:
            return "Unknown"
    
    async def _process_chat_messages(self, chat_element, chat_name: str):
        """Process messages from a specific chat"""
        try:
            # Click on chat to open it
            chat_element.click()
            await asyncio.sleep(1)
            
            # Get message elements
            message_elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                "[data-testid='msg-container']"
            )
            
            # Process recent messages
            for message_element in message_elements[-5:]:  # Last 5 messages
                message_data = await self._extract_message_data(message_element, chat_name)
                
                if message_data and self._is_new_message(message_data):
                    await self._handle_new_message(message_data)
                    
        except Exception as e:
            self.logger.error("Error processing chat messages", chat=chat_name, error=str(e))
    
    async def _extract_message_data(self, message_element, chat_name: str) -> Optional[Dict[str, Any]]:
        """Extract data from message element"""
        try:
            # Extract message text
            text_element = message_element.find_element(
                By.CSS_SELECTOR,
                "[data-testid='conversation-text']"
            )
            message_text = text_element.text
            
            # Extract author (for group messages)
            author = "Unknown"
            try:
                author_element = message_element.find_element(
                    By.CSS_SELECTOR,
                    "[data-testid='quoted-mention']"
                )
                author = author_element.text
            except NoSuchElementException:
                pass
            
            # Extract timestamp
            timestamp = datetime.utcnow()
            
            # Create message ID
            message_id = f"{chat_name}_{int(timestamp.timestamp())}_{hash(message_text)}"
            
            return {
                "message_id": message_id,
                "text": message_text,
                "author": author,
                "chat_name": chat_name,
                "timestamp": timestamp.isoformat(),
                "type": "whatsapp_message",
            }
            
        except Exception as e:
            self.logger.error("Message extraction failed", error=str(e))
            return None
    
    def _is_new_message(self, message_data: Dict[str, Any]) -> bool:
        """Check if message is new"""
        message_id = message_data["message_id"]
        return message_id not in self.message_history
    
    async def _handle_new_message(self, message_data: Dict[str, Any]):
        """Handle new message"""
        try:
            message_id = message_data["message_id"]
            
            # Mark as processed
            self.message_history[message_id] = True
            
            # Save to database
            await self._save_message_to_db(message_data)
            
            # Check if message is a command
            if self.command_handler and await self.command_handler.is_command(message_data["text"]):
                # Handle command and send reply
                command_result = await self.command_handler.handle_command(message_data)
                
                if command_result.get("send_reply") and command_result.get("reply_text"):
                    # Send reply back to WhatsApp
                    await self._send_message({
                        "chat_name": message_data.get("chat_name"),
                        "message": command_result["reply_text"]
                    })
                
                self.logger.info("Command processed", 
                               command=command_result.get("command"), 
                               success=command_result.get("success"))
                
            else:
                # Regular message - send to orchestrator for task analysis
                from core.orchestrator import get_orchestrator
                orchestrator = get_orchestrator()
                await orchestrator.send_message(message_data)
            
            self.logger.info("New message processed", message_id=message_id)
            
        except Exception as e:
            self.logger.error("Message handling failed", error=str(e))
    
    async def _save_message_to_db(self, message_data: Dict[str, Any]):
        """Save message to database"""
        try:
            with get_db_context() as db:
                conversation = Conversation(
                    mensaje=message_data["text"],
                    autor=message_data["author"],
                    timestamp=datetime.fromisoformat(message_data["timestamp"]),
                    grupo_id=message_data["chat_name"],
                    grupo_nombre=message_data["chat_name"],
                    mensaje_id=message_data["message_id"],
                )
                db.add(conversation)
                
        except Exception as e:
            self.logger.error("Database save failed", error=str(e))
    
    async def _save_session(self):
        """Save session data"""
        try:
            session_data = {
                "session_id": self.session_id,
                "is_authenticated": self.is_authenticated,
                "phone_number": self.phone_number,
                "created_at": datetime.utcnow().isoformat(),
                "monitored_groups": self.monitored_groups,
            }
            
            # Save to Redis
            await cache_manager.cache_whatsapp_session(
                self.session_id,
                session_data
            )
            
            # Save to database
            with get_db_context() as db:
                session = WhatsAppSession(
                    session_id=self.session_id,
                    is_authenticated=self.is_authenticated,
                    phone_number=self.phone_number,
                    session_data=json.dumps(session_data),
                )
                db.add(session)
                
        except Exception as e:
            self.logger.error("Session save failed", error=str(e))
    
    async def _save_qr_code(self, qr_code_data: str):
        """Save QR code for web interface"""
        try:
            # Save to Redis for real-time access
            await cache_manager.set(
                f"whatsapp_qr:{self.session_id}",
                qr_code_data,
                expire=300  # 5 minutes
            )
            
        except Exception as e:
            self.logger.error("QR code save failed", error=str(e))
    
    async def _get_qr_code(self) -> Dict[str, Any]:
        """Get current QR code"""
        try:
            qr_data = await cache_manager.get(f"whatsapp_qr:{self.session_id}")
            
            if qr_data:
                return {
                    "success": True,
                    "qr_code": qr_data,
                    "session_id": self.session_id,
                }
            else:
                return {
                    "success": False,
                    "error": "No QR code available",
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _get_whatsapp_status(self) -> Dict[str, Any]:
        """Get WhatsApp connection status"""
        return {
            "success": True,
            "authenticated": self.is_authenticated,
            "session_id": self.session_id,
            "phone_number": self.phone_number,
            "monitored_groups": self.monitored_groups,
            "driver_active": self.driver is not None,
        }
    
    async def _send_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send WhatsApp message"""
        try:
            if not self.is_authenticated or not self.driver:
                return {"success": False, "error": "WhatsApp not authenticated"}
            
            chat_name = data.get("chat_name")
            message_text = data.get("message")
            
            if not chat_name or not message_text:
                return {"success": False, "error": "Missing chat_name or message"}
            
            # Find and open the chat
            chat_found = await self._find_and_open_chat(chat_name)
            if not chat_found:
                return {"success": False, "error": f"Chat '{chat_name}' not found"}
            
            # Send the message
            success = await self._send_message_to_chat(message_text)
            
            if success:
                self.logger.info("Message sent successfully", chat=chat_name, message_length=len(message_text))
                return {
                    "success": True,
                    "message": "Message sent successfully",
                    "chat_name": chat_name
                }
            else:
                return {"success": False, "error": "Failed to send message"}
                
        except Exception as e:
            self.logger.error("Send message failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _get_chats(self) -> Dict[str, Any]:
        """Get list of available chats"""
        # Implementation for getting chat list
        return {"success": False, "error": "Get chats not implemented yet"}
    
    async def _reconnect(self) -> Dict[str, Any]:
        """Reconnect to WhatsApp"""
        try:
            await self._cleanup_agent()
            await self._initialize_agent()
            
            return {
                "success": True,
                "message": "Reconnection successful",
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _find_and_open_chat(self, chat_name: str) -> bool:
        """Find and open a WhatsApp chat by name"""
        try:
            # First, try to find the chat in the chat list
            chat_elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                "[data-testid='chat-list'] > div > div"
            )
            
            for chat_element in chat_elements:
                try:
                    # Get chat name from element
                    name_element = chat_element.find_element(
                        By.CSS_SELECTOR,
                        "[data-testid='conversation-info-header'] span[title]"
                    )
                    
                    if name_element.text == chat_name:
                        # Click on the chat to open it
                        chat_element.click()
                        await asyncio.sleep(1)  # Wait for chat to load
                        return True
                        
                except Exception:
                    continue
            
            # If not found in visible chats, try searching
            return await self._search_and_open_chat(chat_name)
            
        except Exception as e:
            self.logger.error("Find chat failed", chat_name=chat_name, error=str(e))
            return False
    
    async def _search_and_open_chat(self, chat_name: str) -> bool:
        """Search for a chat and open it"""
        try:
            # Click on search button or search box
            search_box = self.driver.find_element(
                By.CSS_SELECTOR,
                "[data-testid='chat-list-search']"
            )
            search_box.click()
            await asyncio.sleep(1)
            
            # Type the chat name
            search_input = self.driver.find_element(
                By.CSS_SELECTOR,
                "input[data-testid='chat-list-search']"
            )
            search_input.clear()
            search_input.send_keys(chat_name)
            await asyncio.sleep(2)  # Wait for search results
            
            # Find and click the first search result
            search_results = self.driver.find_elements(
                By.CSS_SELECTOR,
                "[data-testid='chat-list'] > div > div"
            )
            
            if search_results:
                search_results[0].click()
                await asyncio.sleep(1)
                
                # Clear search
                search_input.clear()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error("Search chat failed", chat_name=chat_name, error=str(e))
            return False
    
    async def _send_message_to_chat(self, message_text: str) -> bool:
        """Send message to currently opened chat"""
        try:
            # Find the message input box
            message_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    "[data-testid='conversation-compose-box-input']"
                ))
            )
            
            # Clear and type the message
            message_box.clear()
            message_box.send_keys(message_text)
            
            # Find and click send button
            send_button = self.driver.find_element(
                By.CSS_SELECTOR,
                "[data-testid='send']"
            )
            send_button.click()
            
            # Wait a moment for message to send
            await asyncio.sleep(1)
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay / 1000)
            
            return True
            
        except TimeoutException:
            self.logger.error("Message box not found - chat may not be open")
            return False
        except Exception as e:
            self.logger.error("Send message to chat failed", error=str(e))
            return False


async def get_whatsapp_status() -> str:
    """Get WhatsApp connection status for health checks"""
    try:
        # This would be called by the main health check
        return "connected"  # or appropriate status
    except Exception:
        return "disconnected"