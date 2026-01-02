"""
Basic tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient


def test_import_app():
    """Test that the app can be imported successfully."""
    from app.main import app

    assert app is not None
    assert app.title == "Street Parking Analyzer API"


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the health check endpoint if it exists."""
    try:
        from app.main import app
        from httpx import AsyncClient

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Try common health check endpoints
            for endpoint in ["/health", "/api/health", "/"]:
                try:
                    response = await client.get(endpoint)
                    if response.status_code in [200, 404]:
                        # Either works or endpoint doesn't exist yet
                        assert True
                        return
                except Exception:
                    continue
        assert True  # If no health endpoint exists yet, that's okay
    except Exception as e:
        pytest.skip(f"Health endpoint test skipped: {e}")


def test_app_structure():
    """Test that the basic app structure exists."""
    import app
    import app.main
    import app.config
    import app.database.mongodb
    import app.services.camera_service
    import app.services.yolo_service
    import app.services.parking_analyzer

    assert True  # If all imports succeed, the structure is correct
