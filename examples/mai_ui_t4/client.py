# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MAI-UI Client Library

A Python client for interacting with the MAI-UI vLLM server.
Provides a simple interface for GUI automation agents.

Usage:
    from client import MAIUIClient
    
    client = MAIUIClient("http://localhost:8000")
    action = client.get_action("screenshot.png", "Click the login button")
    print(action)  # pyautogui.click(500, 300)
    
    # Execute the action
    client.execute_action(action)

Features:
    - Automatic image encoding (base64)
    - Retry logic with exponential backoff
    - Action parsing and execution
    - Support for batch requests
"""

import base64
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """Parsed GUI action from MAI-UI."""
    
    action_type: str  # click, type, scroll, etc.
    raw_text: str  # Original response text
    parameters: dict[str, Any]  # Parsed parameters
    
    def __str__(self) -> str:
        return self.raw_text
    
    @classmethod
    def parse(cls, response_text: str) -> "Action":
        """
        Parse MAI-UI response into structured Action.
        
        Handles formats like:
            - pyautogui.click(500, 300)
            - pyautogui.write("hello")
            - pyautogui.scroll(-3)
            - pyautogui.hotkey("ctrl", "c")
        """
        text = response_text.strip()
        
        # Extract action type and parameters using regex
        # Pattern: pyautogui.action(params) or just action(params)
        pattern = r"(?:pyautogui\.)?(\w+)\((.*)\)"
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return cls(
                action_type="unknown",
                raw_text=text,
                parameters={"raw": text},
            )
        
        action_type = match.group(1)
        params_str = match.group(2).strip()
        
        # Parse parameters based on action type
        parameters = cls._parse_parameters(action_type, params_str)
        
        return cls(
            action_type=action_type,
            raw_text=text,
            parameters=parameters,
        )
    
    @staticmethod
    def _parse_parameters(action_type: str, params_str: str) -> dict[str, Any]:
        """Parse action-specific parameters."""
        params: dict[str, Any] = {"raw": params_str}
        
        if action_type == "click":
            # click(x, y) or click(x, y, button='right')
            coords = re.findall(r"(\d+)", params_str)
            if len(coords) >= 2:
                params["x"] = int(coords[0])
                params["y"] = int(coords[1])
            if "right" in params_str:
                params["button"] = "right"
            elif "middle" in params_str:
                params["button"] = "middle"
            else:
                params["button"] = "left"
                
        elif action_type in ("write", "typewrite"):
            # write("text") or write('text')
            text_match = re.search(r"['\"](.+?)['\"]", params_str)
            if text_match:
                params["text"] = text_match.group(1)
                
        elif action_type == "scroll":
            # scroll(-3) or scroll(5)
            scroll_match = re.search(r"(-?\d+)", params_str)
            if scroll_match:
                params["clicks"] = int(scroll_match.group(1))
                
        elif action_type == "hotkey":
            # hotkey("ctrl", "c")
            keys = re.findall(r"['\"](\w+)['\"]", params_str)
            params["keys"] = keys
            
        elif action_type == "moveTo":
            # moveTo(x, y)
            coords = re.findall(r"(\d+)", params_str)
            if len(coords) >= 2:
                params["x"] = int(coords[0])
                params["y"] = int(coords[1])
                
        elif action_type == "drag":
            # drag(dx, dy)
            coords = re.findall(r"(-?\d+)", params_str)
            if len(coords) >= 2:
                params["dx"] = int(coords[0])
                params["dy"] = int(coords[1])
        
        return params


class MAIUIClient:
    """
    Client for MAI-UI vLLM server.
    
    Provides a high-level interface for GUI automation using
    MAI-UI's vision-language model.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "osunlp/MAI-UI-2B",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize MAI-UI client.
        
        Args:
            base_url: vLLM server URL
            model: Model name (should match server's loaded model)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers["Content-Type"] = "application/json"
    
    def _encode_image(self, image: str | Path | Image.Image) -> str:
        """
        Encode image to base64 data URL.
        
        Args:
            image: Path to image file or PIL Image
        
        Returns:
            Base64 data URL (data:image/png;base64,...)
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            
            # Determine MIME type
            suffix = path.suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_types.get(suffix, "image/png")
            
            with open(path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
                
        elif isinstance(image, Image.Image):
            import io
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_data = base64.b64encode(buffer.getvalue()).decode()
            mime_type = "image/png"
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        return f"data:{mime_type};base64,{image_data}"
    
    def get_action(
        self,
        image: str | Path | Image.Image,
        instruction: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Action:
        """
        Get GUI action for an image and instruction.
        
        Args:
            image: Screenshot (path or PIL Image)
            instruction: Natural language instruction
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0 for deterministic)
        
        Returns:
            Parsed Action object
        
        Example:
            >>> action = client.get_action("screen.png", "Click the submit button")
            >>> print(action)  # pyautogui.click(500, 300)
            >>> print(action.parameters)  # {"x": 500, "y": 300, "button": "left"}
        """
        image_url = self._encode_image(image)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {
                            "type": "text",
                            "text": instruction,
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        
        response_text = response["choices"][0]["message"]["content"]
        return Action.parse(response_text)
    
    def get_action_raw(
        self,
        image: str | Path | Image.Image,
        instruction: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        """
        Get raw action string (no parsing).
        
        Args:
            image: Screenshot (path or PIL Image)
            instruction: Natural language instruction
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
        
        Returns:
            Raw response text from model
        """
        action = self.get_action(image, instruction, max_tokens, temperature)
        return action.raw_text
    
    def batch_get_actions(
        self,
        requests_list: list[dict],
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> list[Action]:
        """
        Get actions for multiple image-instruction pairs.
        
        Note: This sends requests sequentially. For true batching,
        use the offline inference script.
        
        Args:
            requests_list: List of {"image": path, "instruction": str}
            max_tokens: Maximum response tokens per request
            temperature: Sampling temperature
        
        Returns:
            List of Action objects
        """
        results = []
        for req in requests_list:
            action = self.get_action(
                req["image"],
                req["instruction"],
                max_tokens,
                temperature,
            )
            results.append(action)
        return results
    
    def execute_action(self, action: Action, dry_run: bool = False) -> bool:
        """
        Execute a parsed action using pyautogui.
        
        Args:
            action: Parsed Action object
            dry_run: If True, only log action without executing
        
        Returns:
            True if action was executed successfully
        
        Raises:
            ImportError: If pyautogui is not installed
        """
        if dry_run:
            logger.info(f"[DRY RUN] Would execute: {action}")
            return True
        
        try:
            import pyautogui
        except ImportError:
            raise ImportError(
                "pyautogui is required for action execution. "
                "Install with: pip install pyautogui"
            )
        
        action_type = action.action_type
        params = action.parameters
        
        try:
            if action_type == "click":
                x, y = params.get("x"), params.get("y")
                button = params.get("button", "left")
                if x is not None and y is not None:
                    pyautogui.click(x, y, button=button)
                    logger.info(f"Clicked at ({x}, {y}) with {button} button")
                    
            elif action_type in ("write", "typewrite"):
                text = params.get("text", "")
                if text:
                    pyautogui.write(text)
                    logger.info(f"Typed: {text}")
                    
            elif action_type == "scroll":
                clicks = params.get("clicks", 0)
                if clicks:
                    pyautogui.scroll(clicks)
                    logger.info(f"Scrolled: {clicks}")
                    
            elif action_type == "hotkey":
                keys = params.get("keys", [])
                if keys:
                    pyautogui.hotkey(*keys)
                    logger.info(f"Pressed hotkey: {'+'.join(keys)}")
                    
            elif action_type == "moveTo":
                x, y = params.get("x"), params.get("y")
                if x is not None and y is not None:
                    pyautogui.moveTo(x, y)
                    logger.info(f"Moved to ({x}, {y})")
                    
            elif action_type == "drag":
                dx, dy = params.get("dx"), params.get("dy")
                if dx is not None and dy is not None:
                    pyautogui.drag(dx, dy)
                    logger.info(f"Dragged by ({dx}, {dy})")
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if the server is healthy.
        
        Returns:
            True if server is responding
        """
        try:
            response = self._session.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> dict:
        """Make request with retry logic."""
        kwargs.setdefault("timeout", self.timeout)
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._session.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
        
        raise last_error


def main():
    """Example usage of MAI-UI client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MAI-UI Client Example")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--image", required=True, help="Screenshot path")
    parser.add_argument("--instruction", required=True, help="Action instruction")
    parser.add_argument("--execute", action="store_true", help="Execute the action")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no execution)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create client
    client = MAIUIClient(base_url=args.url)
    
    # Check server health
    print(f"🔍 Checking server health at {args.url}...")
    if not client.health_check():
        print("❌ Server is not responding!")
        return
    print("✅ Server is healthy\n")
    
    # Get action
    print(f"📸 Image: {args.image}")
    print(f"📝 Instruction: {args.instruction}\n")
    
    action = client.get_action(args.image, args.instruction)
    
    print("=" * 50)
    print("RESULT")
    print("=" * 50)
    print(f"  Action Type: {action.action_type}")
    print(f"  Raw Output: {action.raw_text}")
    print(f"  Parameters: {action.parameters}")
    print("=" * 50)
    
    # Execute if requested
    if args.execute or args.dry_run:
        print("\n🤖 Executing action...")
        success = client.execute_action(action, dry_run=args.dry_run)
        print(f"{'✅ Success' if success else '❌ Failed'}")


if __name__ == "__main__":
    main()

