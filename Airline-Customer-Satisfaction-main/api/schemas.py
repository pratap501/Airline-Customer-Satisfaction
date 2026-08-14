from pydantic import BaseModel

class PassengerInput(BaseModel):
    Gender: str
    Customer_Type: str
    Age: int
    Type_of_Travel: str
    Class: str
    Flight_Distance: int
    Departure_Delay_in_Minutes: int
    Arrival_Delay_in_Minutes: int

    Seat_comfort: int
    Food_and_drink: int
    Inflight_wifi_service: int
    Inflight_entertainment: int
    Online_support: int
    Ease_of_Online_booking: int
    On_board_service: int
    Leg_room_service: int
    Baggage_handling: int
    Checkin_service: int
    Cleanliness: int
    Online_boarding: int
    Gate_location: int
    Departure_Arrival_time_convenient: int


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
