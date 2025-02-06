import streamlit as st
import pandas as pd
import itertools
import numpy as np

# Set page configuration for mobile view
st.set_page_config(page_title="Table Probability Optimiser", layout="centered")

# Add tab navigation
selected_tab = st.sidebar.radio("Select Page", ["Seat Optimiser", "Light Expected Value"])

if selected_tab == "Seat Optimiser":
    # Title
    st.title("Table Probability Optimiser")

    # Instructions
    st.write("Use this tool to maximise the probabiliy of winning by optimising the seating arrangement")

    # Initialize session state for Tables
    if "Tables" not in st.session_state:
        st.session_state.Tables = [{"Player": 0, "Table": 0, "Optimised": 0} for _ in range(3)]  # Default 3 Tables

    # Initialize session state for overall probabilities
    if "current_prob" not in st.session_state:
        st.session_state.current_prob = 0  # Default value
    if "optimised_prob" not in st.session_state:
        st.session_state.optimised_prob = 0  # Default value

    # Function to optimize probabilities
    def seating_probability():
        tables = len(st.session_state.Tables)
        seats = 7
        open_seats = [7 - table["Player"] for table in st.session_state.Tables]
        total_players = sum(table["Table"] for table in st.session_state.Tables)

        # Generate all possible combinations of seated players per table
        ranges = [list(range(0, open_seats[i] + 1)) for i in range(tables)]
        combinations = list(itertools.product(*ranges))
        
        # Convert to DataFrame
        df = pd.DataFrame(combinations, columns=[f"Table{i+1}" for i in range(tables)])
        
        # Add total column and filter valid combinations
        df["Total"] = df.sum(axis=1)
        filtered_df = df[df["Total"] == total_players].copy()
        
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
            st.session_state.Tables[idx]["Optimised"] = best_solution[idx]

        st.session_state.optimised_prob = round(max_probability,2)

    # Transparent Variables
    NoTables = len(st.session_state.Tables)
    NoSeats = 7
    Open_Seats = [7 - table["Player"] for table in st.session_state.Tables]
    Total_Players = sum(table["Table"] for table in st.session_state.Tables)

    # Section for overall probabilities
    st.subheader("Probability Overview")
    probability_cols = st.columns([1, 1, 1])  # Add one extra column for the button
    probability_cols[1].subheader(f"**Current Probability**: {st.session_state.current_prob}")
    probability_cols[2].subheader(f"**Maximum Probility**: {st.session_state.optimised_prob}")
    probability_cols[1].write(f"**Total Players**: {Total_Players}")
    probability_cols[2].button("Optimise", on_click=seating_probability)

    # Function to add a Table
    def add_Table():
        st.session_state.Tables.append({"Player": 0, "Table": 0, "Optimised": 0})

    # Function to remove a Table
    def remove_Table():
        if len(st.session_state.Tables) > 1:
            st.session_state.Tables.pop()

    # Layout for Tables
    def update_table(idx, key):
        st.session_state.Tables[idx][key] = st.session_state[f"{key}_{idx}"]
        emptyTables = sum(1 for t in st.session_state.Tables if (t["Player"] + t["Table"]) == 0)
        st.session_state.current_prob = 0
        st.session_state.optimised_prob = 0

        if emptyTables < NoTables:
            for t in range(NoTables):
                if (st.session_state.Tables[t]["Player"] + st.session_state.Tables[t]["Table"]) > 0:
                    st.session_state.current_prob += st.session_state.Tables[t]["Table"] / (st.session_state.Tables[t]["Player"] + st.session_state.Tables[t]["Table"])

            st.session_state.current_prob = round(st.session_state.current_prob / (NoTables - emptyTables), 2)

    for idx, Table in enumerate(st.session_state.Tables):
        st.subheader(f"Table {idx + 1}")
        
        with st.container():
            cols = st.columns(3)

            # Number input for Players with on_change callback
            cols[0].number_input(
                f"Opponents (Table {idx + 1})",
                min_value=0,
                max_value=NoSeats-st.session_state.Tables[idx]["Table"],
                value=st.session_state.Tables[idx]["Player"],
                key=f"Player_{idx}",
                on_change=update_table,
                args=(idx, "Player")
            )

            # Number input for Table with on_change callback
            cols[1].number_input(
                f"Team Members (Table {idx + 1})",
                min_value=0,
                max_value=NoSeats-st.session_state.Tables[idx]["Player"],
                value=st.session_state.Tables[idx]["Table"],
                key=f"Table_{idx}",
                on_change=update_table,
                args=(idx, "Table")
            )
            opponents = st.session_state.Tables[idx]["Player"]  # Number of opponent players
            team_members = st.session_state.Tables[idx]["Table"]  # Number of team members
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
                value=st.session_state.Tables[idx]["Optimised"],
                key=f"optimised_{idx}",
                disabled=True
            )

            
            team_members_opt = st.session_state.Tables[idx]["Optimised"]  # Number of team members
            total_filled_opt = opponents + team_members_opt
            empty_seats_opt = 7 - total_filled_opt  # Remaining empty seats

            # Generate seat visualization
            seat_display_opt = " ".join(["🔴"] * opponents + ["🟢"] * team_members_opt + ["⚪"] * empty_seats_opt)

            # Display seats below the inputs
            if st.session_state.optimised_prob!=0:
                #cols[0].write("**Best Seating Arrangement:**")
                cols[2].markdown(f"{seat_display_opt}")

    # Buttons for adding and removing Tables
    st.button("Add Table", on_click=add_Table)
    st.button("Remove Table", on_click=remove_Table)



elif selected_tab == "Light Expected Value":
    # New Page Content
    st.title("Light Expected Value")
    st.subheader("If the EV is above 1, the odds are in you favour")
    
    # Three integer input boxes
    num1 = st.number_input("Top Progressive", min_value=0,value=1000000, step=10000)
    num2 = st.number_input("Second Progressive", min_value=0,value=100000, step=10000)
    num3 = st.number_input("Third Progressive", min_value=0,value=10000, step=1000)
    BetAmount = st.number_input("Bet Amount", min_value=0,value=100, step=50)

    #Display results
    EV = (800*num1+12480*num2+74880*num3+102160*50*BetAmount+204000*10*BetAmount)/(51584880*BetAmount)
    st.subheader(f"Expected value = {round(EV,2)}")





