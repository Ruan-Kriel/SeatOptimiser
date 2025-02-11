import streamlit as st
import pandas as pd
import itertools
import numpy as np
from streamlit_drawable_canvas import st_canvas
import math

def format_currency(number):
    if number < 0:
        return f"-R{abs(number):,}"
    else:
        return f"R{number:,}"

sts=st.session_state

# Set page configuration for mobile view
st.set_page_config(page_title="Table Probability Optimiser", layout="centered")

# Add tab navigation
# Define the tabs
tabs = ["Seat Optimiser", "Poker Side Bet Expected Value", "Poker Expected Value","Blackjack Side Bet Expected Value (Work In Progress)"]

# Conditionally include the last tab
if "total_players" in sts:
    tabs_to_display = tabs  # Show all tabs
else:
    tabs_to_display = ["Seat Optimiser", "Poker Side Bet Expected Value","Blackjack Side Bet Expected Value (Work In Progress)"]  # Exclude the last tab

# Display the radio button with the filtered tabs
selected_tab = st.sidebar.radio("**Select Page**", tabs_to_display)

if selected_tab == "Seat Optimiser":
    # Title
    st.title("Table Probability Optimiser")

    # Instructions
    st.write("Use this tool to maximise the probabiliy of winning by optimising the seating arrangement")

    # Initialize session state for Tables
    if "Tables" not in sts:
        sts.Tables = [{"Player": 0, "Table": 0, "Optimised": 0} for _ in range(3)]  # Default 3 Tables

    # Initialize session state for overall probabilities
    if "current_prob" not in sts:
        sts.current_prob = 0  # Default value
    if "optimised_prob" not in sts:
        sts.optimised_prob = 0  # Default value

    # Function to optimize probabilities
    def seating_probability():
        tables = len(sts.Tables)
        seats = 7
        open_seats = [7 - table["Player"] for table in sts.Tables]
        if (sum(table["Table"] for table in sts.Tables)>0):
            sts.total_players = sum(table["Table"] for table in sts.Tables)
        else:
            sts.total_players = sts.TotalPlayers 


        # Generate all possible combinations of seated players per table
        ranges = [list(range(0, open_seats[i] + 1)) for i in range(tables)]
        combinations = list(itertools.product(*ranges))
        
        # Convert to DataFrame
        df = pd.DataFrame(combinations, columns=[f"Table{i+1}" for i in range(tables)])
        
        # Add total column and filter valid combinations
        df["Total"] = df.sum(axis=1)
        filtered_df = df[df["Total"] == sts.total_players].copy()
        
        # Calculate probability
        row_probabilities = np.zeros(len(filtered_df))
        error_count = np.zeros(len(filtered_df))

        for t in range(tables):
            for i in range(len(filtered_df)):
                denominator = filtered_df.iloc[i, t] + seats - open_seats[t]
                if denominator <= 0:
                    error_count[i] += 1
                else:
                    row_probabilities[i] += filtered_df.iloc[i, t] / denominator
        
        # Adjust probability for tables in play
        valid_tables = tables - error_count
        valid_tables[valid_tables == 0] = 1  # Avoid division by zero
        row_probabilities = row_probabilities / valid_tables
        
        # Find the best solution
        max_index = np.argmax(row_probabilities)
        best_solution = filtered_df.iloc[max_index]
        max_probability = row_probabilities[max_index]

        # Update session state only when optimize button is clicked
        for idx in range(len(best_solution)-1):
            sts.Tables[idx]["Optimised"] = best_solution[idx]

        sts.optimised_prob = round(max_probability,2)
   


    # Setting total player box to equal selected players
    if (sum(table["Table"] for table in sts.Tables)>0):
        Total_Players = sum(table["Table"] for table in sts.Tables)
        sts.TotalPlayers = Total_Players

    if "total_players" not in sts:
        sts.total_players=0

    st.subheader("Probability Overview")
    probability_cols = st.columns([1, 1])  # Add one extra column for the button
    probability_cols[0].number_input("**Total Team Members**",min_value=0,key="TotalPlayers",value=sts.total_players)
    probability_cols[0].write("If team members entered below total team members cannot be changed")
    probability_cols[1].subheader(f"**Current Probability**: {sts.current_prob}")
    probability_cols[1].subheader(f"**Maximum Probility**: {sts.optimised_prob}")
    probability_cols[1].button("Optimise", on_click=seating_probability,type="primary")

     # Transparent Variables
    NoTables = len(sts.Tables)
    NoSeats = 7
    Open_Seats = [7 - table["Player"] for table in sts.Tables]
    if (sum(table["Table"] for table in sts.Tables)>0):
        Total_Players = sum(table["Table"] for table in sts.Tables)
    else:
        Total_Players = sts.TotalPlayers 

        
    # Function to add a Table
    def add_Table():
        sts.Tables.append({"Player": 0, "Table": 0, "Optimised": 0})

    # Function to remove a Table
    def remove_Table():
        if len(sts.Tables) > 1:
            sts.Tables.pop()

    # Layout for Tables
    def update_table(idx, key):
        sts.Tables[idx][key] = sts[f"{key}_{idx}"]
        emptyTables = sum(1 for t in sts.Tables if (t["Player"] + t["Table"]) == 0)
        sts.current_prob = 0
        sts.optimised_prob = 0

        if emptyTables < NoTables:
            for t in range(NoTables):
                if (sts.Tables[t]["Player"] + sts.Tables[t]["Table"]) > 0:
                    sts.current_prob += sts.Tables[t]["Table"] / (sts.Tables[t]["Player"] + sts.Tables[t]["Table"])

            sts.current_prob = round(sts.current_prob / (NoTables - emptyTables), 2)

    for idx, Table in enumerate(sts.Tables):
        st.subheader(f"Table {idx + 1}")
        
        with st.container():
            cols = st.columns(3)

            # Number input for Players with on_change callback
            cols[0].number_input(
                f"Opponents (Table {idx + 1})",
                min_value=0,
                max_value=NoSeats-sts.Tables[idx]["Table"],
                value=sts.Tables[idx]["Player"],
                key=f"Player_{idx}",
                on_change=update_table,
                args=(idx, "Player")
            )

            # Number input for Table with on_change callback
            cols[1].number_input(
                f"Team Members (Table {idx + 1})",
                min_value=0,
                max_value=NoSeats-sts.Tables[idx]["Player"],
                value=sts.Tables[idx]["Table"],
                key=f"Table_{idx}",
                on_change=update_table,
                args=(idx, "Table")
            )
            opponents = sts.Tables[idx]["Player"]  # Number of opponent players
            team_members = sts.Tables[idx]["Table"]  # Number of team members
            total_filled = opponents + team_members
            empty_seats = 7 - total_filled  # Remaining empty seats

            # Generate seat visualization
            seat_display = " ".join(["🔴"] * opponents + ["🟢"] * team_members + ["⚪"] * empty_seats)

           # Display seats below the inputs
            #cols[0].write("**Current Seating Arrangement:**")
            cols[1].markdown(f"{seat_display}")


            # Text input for Optimised (disabled to prevent unwanted changes)
            cols[2].text_input(
                f"Optimised (Table {idx + 1})",
                value=sts.Tables[idx]["Optimised"],
                key=f"optimised_{idx}",
                disabled=True
            )

            
            team_members_opt = sts.Tables[idx]["Optimised"]  # Number of team members
            total_filled_opt = opponents + team_members_opt
            empty_seats_opt = 7 - total_filled_opt  # Remaining empty seats

            # Generate seat visualization
            seat_display_opt = " ".join(["🔴"] * opponents + ["🟢"] * team_members_opt + ["⚪"] * empty_seats_opt)

            # Display seats below the inputs
            if sts.optimised_prob!=0:
                #cols[0].write("**Best Seating Arrangement:**")
                cols[2].markdown(f"{seat_display_opt}")

    # Buttons for adding and removing Tables
    st.button("Add Table", on_click=add_Table)
    st.button("Remove Table", on_click=remove_Table)



elif selected_tab == "Poker Side Bet Expected Value":
    # New Page Content
    st.title("Poker Side Bet Expected Value")
    st.subheader("If the EV is above 1, the odds are in you favour")
    
    if "num1" not in sts:
        sts.num1 = 1000000
        sts.num2 = 100000
        sts.num3 = 10000

    # Three integer input boxes
    sts.num1 = st.number_input("Top Progressive", min_value=0,value= sts.num1, step=10000)
    sts.num2 = st.number_input("Second Progressive", min_value=0,value= sts.num2, step=10000)
    sts.num3 = st.number_input("Third Progressive", min_value=0,value= sts.num3, step=1000)
    BetAmount = st.number_input("Bet Amount", min_value=0,value=100, step=50)

    #Display results
    sts.EV_Poker_Screen = (800*sts.num1+12480*sts.num2+74880*sts.num3+102160*50*BetAmount+204000*10*BetAmount)/(51584880*BetAmount)
    sts.EV_Poker_Screen_Reduced = (0*sts.num1+0*sts.num2+74880*sts.num3+102160*50*BetAmount+204000*10*BetAmount)/(51584880*BetAmount)
    st.subheader(f"Expected value = **{round( sts.EV_Poker_Screen,2)}**")

elif selected_tab == "Blackjack Side Bet Expected Value (Work In Progress)":
    # New Page Content
    st.title("Blackjack Side Bet Expected Value")
    st.subheader("If the EV is above 1, the odds are in you favour")
    
    # Three integer input boxes
    num1 = st.number_input("Top Progressive", min_value=0,value=1000000, step=10000)
    num2 = st.number_input("Second Progressive", min_value=0,value=100000, step=10000)
    num3 = st.number_input("Third Progressive", min_value=0,value=10000, step=1000)
    BetAmount = st.number_input("Bet Amount", min_value=0,value=100, step=50)

    #Display results
    sts.EV_BJ_Screen = (800*num1+12480*num2+74880*num3+102160*50*BetAmount+204000*10*BetAmount)/(51584880*BetAmount)
    st.subheader(f"Expected value = **{round( sts.EV_BJ_Screen,2)}**")

elif selected_tab == "Poker Expected Value":
    st.title("Poker Expected Value")
    #sts.Bet=st.number_input("Per Hand Bet amount:",min_value=0,step=50,value=200)
    #sts.Lights_Bet=st.number_input("Per Hand Lights Bet amount:",min_value=0,step=50,value=100)
    sts.TotalBets = st.number_input("Bets per hour:",min_value=0,step=500,value=5000)
    sts.TotalLightBets = st.number_input("Light Bets per hour:",min_value=0,step=500,value=3000)
    sts.CurrentMystery = st.number_input("Poker Mystery Jackpot:",min_value=0,step=5000,value=170000)
    st.write(f"The mystery increases at a rate of **{format_currency(round(6000*(sts.total_players+sum(table["Player"] for table in sts.Tables))/20))}** per hour based on current assumptions and a player count of **{sum(table["Player"] for table in sts.Tables)+sts.total_players}**")
    sts.lightRate = 6000*(sts.total_players+sum(table["Player"] for table in sts.Tables))/20
    sts.timeRemaining = (200000-sts.CurrentMystery)/sts.lightRate
    sts.slice = math.ceil(sts.timeRemaining)
    st.write(f"It would therefore take **{round(sts.timeRemaining,2)}** hours to reach R200,000")
    sts.dist=st.selectbox("Please select the assumed mystery drop distribution below:",["Increasing","Flat"])
    sts.BigWins = st.checkbox("Remove four of a kind and straight flush:")
    #Will go go casino and figure out the rate of increase for a given amount of players. Lets say R6000 per hour for 20 players.
    

    if sts.dist=="Flat":
        sts.distribution = np.ones(sts.slice)/sts.slice
    else:
        # Calculate the distribution
        sts.distribution = np.arange(1, sts.slice + 1) / sts.slice

        # Normalize the distribution
        sts.distribution = sts.distribution / np.sum(sts.distribution)
    
    PokerEv = np.empty(sts.slice)
    TotalSpend = np.empty(sts.slice)
    for t in range(0,sts.slice):
        
        PokerEv[t] = sts.distribution[t]*(sts.optimised_prob/sts.total_players*(sts.CurrentMystery+(200000-sts.CurrentMystery)*(t+1)/sts.slice)
        +((sts.EV_Poker_Screen*(1-sts.BigWins)+sts.EV_Poker_Screen_Reduced*(sts.BigWins)-1)*sts.TotalLightBets+((1-sts.BigWins)*0.9704+sts.BigWins*0.95176-1)*sts.TotalBets-200)*sts.timeRemaining/sts.slice*(t+1))

        TotalSpend[t] = sts.distribution[t]*(sts.TotalBets+sts.TotalLightBets)*sts.timeRemaining/sts.slice*(t+1)

    print(TotalSpend)
    print(np.sum(TotalSpend)*sts.total_players)
    print(np.sum(PokerEv)/(np.sum(TotalSpend)*sts.total_players))

    st.write(f"The expected value per player is **{format_currency(round(np.sum(PokerEv)))}**")
    st.write(f"The total expected value is **{format_currency(round(np.sum(PokerEv)*sts.total_players))}**")

    st.subheader("Extra Information:")
    st.write(f"Each hour of normal poker play contributes **{format_currency(round(((1-sts.BigWins)*0.9704+sts.BigWins*0.95176-1)*sts.TotalBets))}** per player")
    st.write(f"Each hour of playing the light for progressive contributes **{format_currency(round((sts.EV_Poker_Screen*(1-sts.BigWins)+sts.EV_Poker_Screen_Reduced*(sts.BigWins)-1)*sts.TotalLightBets))}** per player")
    st.write(f"Each hour of playing the light for mystery contributes on average **{format_currency(round((sts.optimised_prob/sts.total_players*(200000+sts.CurrentMystery)/2)/sts.timeRemaining))}** per player")
    st.write(f"Wage for each hour of play contributes **-R200** per player")
    st.write(f"Current conditions reflect a total return of **{round(np.sum(PokerEv)/(np.sum(TotalSpend)*sts.total_players)*100,2)}%** over the long run")


        


    

