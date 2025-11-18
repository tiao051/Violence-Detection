from fastapi import APIRouter, status

# Define the router instance
router = APIRouter(
    tags=["Cameras"],
)

@router.get("/cameras", status_code=status.HTTP_200_OK)
async def get_camera_list():
    """
    Retrieves the list of available cameras.
    NOTE: Currently returns an empty list. Implementation logic goes here.
    """
    # Later: You will add logic here to fetch cameras from MediaMTX or a database.
    return []