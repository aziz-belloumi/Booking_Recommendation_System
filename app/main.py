from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from recommendation_service import RecommendationService
import traceback
from schema import RecommendRequest

app = FastAPI()


try:
    recommendation_service = RecommendationService()
except Exception as e:
    print(f"Failed to initialize recommendation service: {str(e)}")
    recommendation_service = None



@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    """Get slot recommendations"""
    try:
        if recommendation_service is None:
            raise HTTPException(status_code=500, detail="Service Not Initialized")

        recommendations = recommendation_service.recommend_slots(
            user_id=req.user_id,
            purpose=req.purpose,
            attendees=req.attendees,
            target_date=req.target_date,
            target_hours=req.target_hours,
            top_k=req.top_k
        )

        return {
            'success': True,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
        }

    except Exception as e:
        print(f"Error in recommendation endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"*internal Server Error: {str(e)}")


@app.get("/room")
def get_room(room_id: Optional[str] = Query(None)):
    try:
        if recommendation_service is None:
            raise HTTPException(status_code=500, detail="Service not initialized")

        room_info = recommendation_service.get_room_info(room_id)

        return {
            'success': True,
            'rooms': room_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}/preferences")
def get_user_preferences(user_id: int):
    try:
        if recommendation_service is None:
            raise HTTPException(status_code=500, detail="Service not initialized")

        preferences = recommendation_service.get_user_preferences(user_id)

        return {
            'success': True,
            'user_id': user_id,
            'preferred_rooms': preferences
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))