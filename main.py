import os
import streamlit as st
import pandas as pd
from agno.agent import Agent
from agno.models.google import Gemini

# ------------------------
# Set your API key
# ------------------------
os.environ["AGNO_API_KEY"] = "Your_Key"  # Replace with your actual key

# ------------------------
# Load mock datasets
# ------------------------
flights_df = pd.read_csv("flights.csv")
hotels_df = pd.read_csv("hotels.csv")

# ------------------------
# Initialize Agno Agents with API Key
# ------------------------
flight_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-001", api_key=os.environ["AGNO_API_KEY"]),
    instructions="You are a flight search assistant. Suggest the top 3 flights from the dataset based on user input."
)

hotel_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-001", api_key=os.environ["AGNO_API_KEY"]),
    instructions="You are a hotel search assistant. Suggest the top 3 hotels based on city and budget."
)

itinerary_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-001", api_key=os.environ["AGNO_API_KEY"]),
    instructions="You are a travel itinerary planner. Create a 3-day itinerary for the destination."
)

booking_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-001", api_key=os.environ["AGNO_API_KEY"]),
    instructions="You are a booking assistant. Confirm the trip and generate a booking summary."
)

# ------------------------
# Streamlit UI
# ------------------------
st.title("✈️ Travel Booking Assistant")

origin = st.text_input("Origin", "Delhi")
destination = st.text_input("Destination", "Paris")
date = st.date_input("Travel Date")
budget = st.number_input("Hotel Budget per Night (USD)", 50, 500, 150)
style = st.selectbox("Travel Style", ["Leisure", "Adventure", "Cultural", "Romantic"])

if st.button("Search & Book"):

    # --- Flights ---
    flights_filtered = flights_df[
        (flights_df.origin == origin) &
        (flights_df.destination == destination)
    ]
    flight_prompt = f"User wants flights from {origin} to {destination} on {date}. Show top 3 options."
    flight_response = flight_agent.run(flight_prompt)

    st.subheader("Available Flights ✈️")
    st.dataframe(flights_filtered)
    st.write("Agent Suggestion:", flight_response.content)

    # --- Hotels ---
    hotels_filtered = hotels_df[
        (hotels_df.city == destination) &
        (hotels_df.price_per_night <= budget)
    ]
    hotel_prompt = f"User wants hotels in {destination} under {budget} USD/night. Suggest top 3."
    hotel_response = hotel_agent.run(hotel_prompt)

    st.subheader("Available Hotels 🏨")
    st.dataframe(hotels_filtered)
    st.write("Agent Suggestion:", hotel_response.content)

    # --- Itinerary ---
    itinerary_prompt = f"Plan a {style} 3-day trip to {destination}."
    itinerary_response = itinerary_agent.run(itinerary_prompt)

    st.subheader("Suggested Itinerary 📅")
    st.write(itinerary_response.content)

    # --- Booking Confirmation ---
    booking_prompt = f"Confirm booking for {origin} -> {destination}, flight, hotel, itinerary."
    booking_response = booking_agent.run(booking_prompt)

    st.success("Booking Confirmed ✅")
    st.write(booking_response.content)
