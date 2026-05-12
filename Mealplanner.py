import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD

# --- AI DATASET ---
def get_meal_data():
    # Data structure: Name, Calories, Protein, Cost, Ingredients, Recipe, Image
    data = [
        ["Oatmeal and Peanut Butter", 400, 15, 5, "Oats, Peanut Butter, Milk", "1. Boil 1 cup water/milk. \n2. Add oats, cook 5 mins. \n3. Stir in peanut butter.", "https://images.unsplash.com/photo-1517673132405-a56a62b18caf?w=400"],
        ["Chicken and Rice Bowl", 650, 40, 12, "Chicken, Rice, Broccoli", "1. Pan-fry chicken. \n2. Steam rice. \n3. Combine with soy sauce.", "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400"],
        ["Lentil Soup (Dal)", 350, 18, 4, "Lentils, Garlic, Cumin", "1. Boil lentils. \n2. Sauté garlic/cumin. \n3. Mix.", "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=400"],
        ["Greek Yogurt Granola", 300, 20, 8, "Yogurt, Granola, Honey", "1. Bowl yogurt. \n2. Top with granola.", "https://images.unsplash.com/photo-1488477181946-6428a0291777?w=400"],
        ["Tuna Pasta Salad", 500, 35, 10, "Tuna, Pasta, Mayo", "1. Boil pasta. \n2. Mix with tuna/mayo.", "https://images.unsplash.com/photo-1551183053-bf91a1d81141?w=400"],
        ["Egg and Cheese Toast", 350, 20, 6, "Eggs, Bread, Cheese", "1. Scramble eggs. \n2. Melt cheese on top. \n3. Serve on toast.", "https://images.unsplash.com/photo-1525351484163-7529414344d8?w=400"],
        ["Chickpea Curry", 450, 15, 5, "Chickpeas, Onion, Tomato", "1. Sauté onion. \n2. Add chickpeas/tomato. \n3. Simmer.", "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400"],
        ["Beef Broccoli Stir-fry", 700, 45, 18, "Beef, Broccoli, Soy Sauce", "1. Sear beef. \n2. Stir-fry with broccoli.", "https://images.unsplash.com/photo-1512058560564-64047bc34e02?w=400"],
        ["Protein Shake Banana", 250, 25, 7, "Protein Powder, Banana, Milk", "1. Blend all items.", "https://images.unsplash.com/photo-1553530666-ba11a7da3888?w=400"],
        ["Black Bean Tacos", 550, 22, 9, "Beans, Tortilla, Salsa", "1. Heat beans. \n2. Fill tortillas.", "https://images.unsplash.com/photo-1551504734-5ee1c4a1479b?w=400"],
        ["Avocado Toast", 320, 8, 15, "Avocado, Bread, Lemon", "1. Mash avocado. \n2. Spread on toast.", "https://images.unsplash.com/photo-1525351484163-7529414344d8?w=400"],
        ["Salmon and Asparagus", 600, 45, 25, "Salmon, Asparagus", "1. Roast at 200C for 15 mins.", "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=400"],
        ["Quinoa Power Bowl", 480, 18, 14, "Quinoa, Sweet Potato, Kale", "1. Cook quinoa. \n2. Mix with veg.", "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400"],
        ["Turkey Sandwich", 450, 30, 11, "Turkey, Bread, Tomato", "1. Layer turkey and veg.", "https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=400"],
        ["Spinach Omelette", 300, 22, 7, "Eggs, Spinach", "1. Cook eggs with spinach.", "https://images.unsplash.com/photo-1510629954389-c1e0da47d4ec?w=400"],
        ["Pesto Pasta", 550, 15, 13, "Pasta, Pesto, Parmesan", "1. Mix boiled pasta with pesto.", "https://images.unsplash.com/photo-1473093226795-af9932fe5856?w=400"],
        ["Chicken Wrap", 520, 35, 12, "Chicken, Tortilla, Pepper", "1. Grill chicken. \n2. Wrap with veg.", "https://images.unsplash.com/photo-1626700051175-65686c483728?w=400"],
        ["Sweet Potato Chili", 400, 14, 8, "Sweet Potato, Beans, Tomato", "1. Simmer all items 25 mins.", "https://images.unsplash.com/photo-1543339308-43e59d6b73a6?w=400"],
        ["Cottage Cheese Fruit", 220, 25, 9, "Cottage Cheese, Apple", "1. Mix cheese and fruit.", "https://images.unsplash.com/photo-1559181567-c3190cb9959b?w=400"],
        ["Shrimp Stir-fry", 450, 38, 20, "Shrimp, Peas, Rice", "1. Sauté shrimp and peas.", "https://images.unsplash.com/photo-1512058560564-64047bc34e02?w=400"],
        ["Hummus Pita", 350, 12, 10, "Hummus, Pita, Carrots", "1. Serve hummus with pita/veg.", "https://images.unsplash.com/photo-1577906030551-879417240409?w=400"],
        ["Berry Smoothie Bowl", 310, 10, 16, "Berries, Banana, Milk", "1. Blend thick. \n2. Bowl.", "https://images.unsplash.com/photo-1590301157890-4810ed352733?w=400"],
        ["Peanut Noodles", 580, 18, 9, "Noodles, Peanut Butter", "1. Toss noodles in sauce.", "https://images.unsplash.com/photo-1585032226651-759b368d7246?w=400"]
    ]
    return pd.DataFrame(data, columns=["Name", "Calories", "Protein", "Cost", "Ingredients", "Recipe", "Image"])

def solve_with_inventory(df, target_cals, weekly_budget, inventory):
    prob = LpProblem("Inventory_Optimization", LpMaximize)
    meal_vars = LpVariable.dicts("Meal", df.index, 0, 1, cat='Binary')
    
    def get_score(row):
        score = row['Protein']
        for item in inventory:
            if item.lower().strip() in row['Ingredients'].lower():
                score += 50 
        return score

    prob += lpSum([get_score(df.loc[i]) * meal_vars[i] for i in df.index])
    prob += lpSum([df.loc[i, 'Cost'] * meal_vars[i] for i in df.index]) <= weekly_budget
    prob += lpSum([df.loc[i, 'Calories'] * meal_vars[i] for i in df.index]) >= target_cals * 0.85
    prob += lpSum([meal_vars[i] for i in df.index]) == 21
    
    prob.solve(PULP_CBC_CMD(msg=0))
    if value(prob.objective):
        return [i for i in df.index if value(meal_vars[i]) == 1]
    return []

# --- STREAMLIT UI ---
st.set_page_config(page_title="Math Stack AI", layout="wide")
st.title("The Math Stack: Inventory-Aware AI Meal Planner")

with st.sidebar:
    st.header("1. Personal Stats")
    age = st.slider("Age", 14, 22, 18)
    gender = st.selectbox("Gender", ["Male", "Female"])
    activity = st.selectbox("Lifestyle", ["Sedentary", "Active"])
    
    st.header("2. Finances")
    budget = st.number_input("Weekly Budget (AED)", 200, 2000, 500)
    
    st.header("3. Inventory at Home")
    user_inv = st.text_input("What's in your fridge? (e.g. Chicken, Eggs, Rice)")
    inventory_list = user_inv.split(",") if user_inv else []

df_meals = get_meal_data()
daily_req = 2000 

if st.button("Generate Plan Based on My Ingredients"):
    indices = solve_with_inventory(df_meals, daily_req*7, budget, inventory_list)
    if not indices:
        st.error("Budget too low or calorie target unreachable.")
    else:
        selected_df = df_meals.loc[indices].sample(frac=1).reset_index(drop=True)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for i, day in enumerate(days):
            st.subheader(day)
            cols = st.columns(3)
            for j in range(3):
                meal = selected_df.iloc[i*3 + j]
                with cols[j]:
                    st.image(meal['Image'], use_container_width=True)
                    if st.button(meal['Name'], key=f"{day}_{j}"):
                        st.info(f"**Step-by-Step Recipe:** \n{meal['Recipe']}")
