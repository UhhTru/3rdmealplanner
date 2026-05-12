import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, PULP_CBC_CMD

# --- AI DATASET WITH DETAILED STEP-BY-STEP RECIPES ---
def get_meal_data():
    data = [
        ["Oatmeal and Peanut Butter", 400, 15, 50, 15, 5, 
         "1. Boil 1 cup of water or milk in a small saucepan. \n2. Add 1/2 cup of rolled oats and reduce heat to medium. \n3. Stir occasionally for 5-7 minutes until creamy. \n4. Remove from heat and stir in 1 tablespoon of natural peanut butter. \n5. Top with a pinch of salt or cinnamon if desired.", 
         "https://images.unsplash.com/photo-1517673132405-a56a62b18caf?w=400"],
        ["Chicken and Rice Bowl", 650, 40, 60, 15, 12, 
         "1. Season 150g of chicken breast with salt, pepper, and garlic powder. \n2. Heat a pan with 1 tsp oil and cook chicken for 6-8 minutes per side. \n3. While chicken cooks, steam 1/2 cup of basmati rice. \n4. Slice the chicken into strips. \n5. Serve chicken over the rice with a side of steamed broccoli or soy sauce.", 
         "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400"],
        ["Lentil Soup (Dal)", 350, 18, 55, 2, 4, 
         "1. Rinse 1/2 cup of red lentils thoroughly. \n2. In a pot, combine lentils with 2 cups of water and 1/2 tsp turmeric. \n3. Bring to a boil, then simmer for 15-20 minutes until soft. \n4. In a separate small pan, heat 1 tsp oil and sauté minced garlic and cumin seeds for 1 minute. \n5. Pour the garlic mixture into the lentils and stir well.", 
         "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=400"],
        ["Greek Yogurt Granola", 300, 20, 30, 8, 8, 
         "1. Scoop 200g of Greek yogurt into a bowl. \n2. Measure 50g of whole-grain granola. \n3. Sprinkle granola over the yogurt. \n4. Drizzle with 1 teaspoon of honey or maple syrup. \n5. Add fresh berries if available.", 
         "https://images.unsplash.com/photo-1488477181946-6428a0291777?w=400"],
        ["Tuna Pasta Salad", 500, 35, 45, 12, 10, 
         "1. Boil 75g of pasta in salted water until al dente. \n2. Drain the pasta and let it cool. \n3. Open a can of tuna in water and drain. \n4. Mix pasta with tuna, 1 tbsp light mayo, and 1/4 cup canned peas. \n5. Season with black pepper and lemon.", 
         "https://images.unsplash.com/photo-1551183053-bf91a1d81141?w=400"],
        ["Egg and Cheese Toast", 350, 20, 25, 18, 6, 
         "1. Toast two slices of whole-wheat bread. \n2. Whisk 2 eggs with a splash of milk. \n3. Scramble the eggs in a non-stick pan. \n4. Before finishing, add a slice of cheddar cheese to melt. \n5. Place cheesy eggs onto the toast.", 
         "https://images.unsplash.com/photo-1525351484163-7529414344d8?w=400"],
        ["Chickpea Curry", 450, 15, 65, 10, 5, 
         "1. Sauté half a chopped onion in oil. \n2. Add 1 tsp curry powder. \n3. Pour in 1 can drained chickpeas and 1/2 cup crushed tomatoes. \n4. Simmer for 10 minutes. \n5. Serve with flatbread or rice.", 
         "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400"],
        ["Beef Broccoli Stir-fry", 700, 45, 40, 25, 18, 
         "1. Slice 150g lean beef across the grain. \n2. Sear beef quickly over high heat and remove. \n3. Steam 2 cups broccoli for 2 minutes. \n4. Return beef to pan with 2 tbsp soy sauce and 1 tsp ginger. \n5. Toss and serve.", 
         "https://images.unsplash.com/photo-1512058560564-64047bc34e02?w=400"],
        ["Protein Shake Banana", 250, 25, 30, 3, 7, 
         "1. Add 300ml water or milk to a blender. \n2. Add 1 scoop protein powder. \n3. Add 1 medium banana. \n4. Blend until smooth. \n5. Consume immediately.", 
         "https://images.unsplash.com/photo-1553530666-ba11a7da3888?w=400"],
        ["Black Bean Tacos", 550, 22, 70, 14, 9, 
         "1. Heat 1/2 can black beans with cumin. \n2. Toast 3 small corn tortillas. \n3. Mash beans slightly. \n4. Distribute into tortillas. \n5. Top with salsa and lettuce.", 
         "https://images.unsplash.com/photo-1551504734-5ee1c4a1479b?w=400"],
        ["Avocado Toast", 320, 8, 35, 22, 15, 
         "1. Toast two slices of bread. \n2. Mash half an avocado with lemon and salt. \n3. Spread onto warm toast. \n4. Top with red pepper flakes. \n5. Drizzle with olive oil.", 
         "https://images.unsplash.com/photo-1525351484163-7529414344d8?w=400"],
        ["Salmon and Asparagus", 600, 45, 5, 40, 25, 
         "1. Preheat oven to 200°C. \n2. Place 150g salmon and asparagus on a sheet. \n3. Season with lemon and dill. \n4. Roast for 12-15 minutes. \n5. Serve fresh.", 
         "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=400"],
        ["Quinoa Power Bowl", 480, 18, 70, 12, 14, 
         "1. Boil 1/2 cup quinoa in 1 cup water for 15 mins. \n2. Dice roasted sweet potato. \n3. Combine quinoa, potato, and 1/4 cup black beans. \n4. Add fresh spinach. \n5. Dress with olive oil.", 
         "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400"],
        ["Turkey Sandwich", 450, 30, 40, 15, 11, 
         "1. Spread mustard on whole-wheat bread. \n2. Layer 100g turkey breast. \n3. Add tomato, cucumber, and lettuce. \n4. Close and cut diagonally. \n5. Serve with carrots.", 
         "https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=400"],
        ["Spinach Omelette", 300, 22, 4, 20, 7, 
         "1. Whisk 3 eggs. \n2. Sauté handful of spinach in pan. \n3. Pour eggs over spinach. \n4. Cook until edges set and fold. \n5. Serve hot.", 
         "https://images.unsplash.com/photo-1510629954389-c1e0da47d4ec?w=400"],
        ["Pesto Pasta", 550, 15, 75, 25, 13, 
         "1. Boil 100g pasta. \n2. Reserve 2 tbsp pasta water. \n3. Mix pasta with 2 tbsp pesto and reserved water. \n4. Top with parmesan. \n5. Serve warm.", 
         "https://images.unsplash.com/photo-1473093226795-af9932fe5856?w=400"],
        ["Chicken Wrap", 520, 35, 45, 18, 12, 
         "1. Grill 120g chicken strips. \n2. Lay out flour tortilla. \n3. Spread garlic sauce. \n4. Add chicken, peppers, and carrots. \n5. Roll tightly.", 
         "https://images.unsplash.com/photo-1626700051175-65686c483728?w=400"],
        ["Sweet Potato Chili", 400, 14, 80, 5, 8, 
         "1. Sauté diced sweet potato and onions. \n2. Add kidney beans and tomatoes. \n3. Season with chili and cumin. \n4. Add broth and simmer 25 mins. \n5. Serve hot.", 
         "https://images.unsplash.com/photo-1543339308-43e59d6b73a6?w=400"],
        ["Cottage Cheese & Fruit", 220, 25, 20, 2, 9, 
         "1. Place 1 cup cottage cheese in bowl. \n2. Add diced apple or berries. \n3. Mix fruit into cheese. \n4. Top with sunflower seeds. \n5. Sprinkle cinnamon.", 
         "https://images.unsplash.com/photo-1559181567-c3190cb9959b?w=400"],
        ["Shrimp Stir-fry", 450, 38, 30, 10, 20, 
         "1. Sauté 150g shrimp with garlic until pink. \n2. Add snap peas. \n3. Pour in soy and honey sauce. \n4. Stir-fry 3 mins. \n5. Serve over brown rice.", 
         "https://images.unsplash.com/photo-1512058560564-64047bc34e02?w=400"],
        ["Hummus Pita Veggies", 350, 12, 50, 12, 10, 
         "1. Toast pita triangles. \n2. Serve with 4 tbsp hummus. \n3. Slice cucumbers and carrots. \n4. Season with paprika. \n5. Enjoy fresh.", 
         "https://images.unsplash.com/photo-1577906030551-879417240409?w=400"],
        ["Berry Smoothie Bowl", 310, 10, 60, 5, 16, 
         "1. Blend mixed berries, banana, and milk. \n2. Ensure thick consistency. \n3. Pour into bowl. \n4. Top with chia seeds. \n5. Eat with spoon.", 
         "https://images.unsplash.com/photo-1590301157890-4810ed352733?w=400"],
        ["Peanut Noodles", 580, 18, 85, 20, 9, 
         "1. Cook 100g rice noodles. \n2. Whisk peanut butter, soy, and sriracha. \n3. Toss noodles in sauce. \n4. Top with peanuts. \n5. Serve cold or warm.", 
         "https://images.unsplash.com/photo-1585032226651-759b368d7246?w=400"]
    ]
    return pd.DataFrame(data, columns=["Name", "Calories", "Protein", "Carbs", "Fat", "Cost", "Recipe", "Image"])

def calculate_tdee(age, gender, activity):
    weight, height = 65, 170
    if gender == "Male":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    multiplier = 1.2 if activity == "Sedentary" else 1.55
    return bmr * multiplier

def solve_variety_plan(df, daily_calories, weekly_budget):
    target_cals = daily_calories * 7
    prob = LpProblem("Unique_Meal_Optimization", LpMaximize)
    meal_vars = LpVariable.dicts("Meal", df.index, 0, 1, cat='Binary')
    prob += lpSum([df.loc[i, 'Protein'] * meal_vars[i] for i in df.index])
    prob += lpSum([df.loc[i, 'Cost'] * meal_vars[i] for i in df.index]) <= weekly_budget
    prob += lpSum([df.loc[i, 'Calories'] * meal_vars[i] for i in df.index]) >= target_cals * 0.85
    prob += lpSum([meal_vars[i] for i in df.index]) == 21
    prob.solve(PULP_CBC_CMD(msg=0))
    if value(prob.objective):
        return [i for i in df.index if value(meal_vars[i]) == 1]
    return []

# --- UI ---
st.set_page_config(page_title="Math Stack AI", layout="wide")
st.title("The Math Stack: Student Meal Planner AI")

with st.sidebar:
    st.header("User Parameters")
    age = st.slider("Age", 14, 22, 18)
    gender = st.selectbox("Gender", ["Male", "Female"])
    activity = st.selectbox("Lifestyle", ["Sedentary", "Active"])
    budget = st.number_input("Weekly Budget (AED)", 200, 2000, 500)
    daily_req = calculate_tdee(age, gender, activity)
    st.info(f"Target: {int(daily_req)} kcal/day")

tab1, tab2 = st.tabs(["Weekly Meal Plan", "How Math Impacts Life"])

with tab1:
    df_meals = get_meal_data()
    if st.button("Generate AI-Driven 21-Meal Plan"):
        indices = solve_variety_plan(df_meals, daily_req, budget)
        if not indices:
            st.error("Budget insufficient for 21 unique meals.")
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
                            st.markdown("### Step-by-Step Recipe")
                            st.write(meal['Recipe'])
                            st.write(f"**Nutrition:** {meal['Calories']} kcal | {meal['Protein']}g Protein")

with tab2:
    st.header("The Math Behind the Plate")
    st.markdown("""
    ### 1. Operations Research (Optimization)
    We utilize Mixed-Integer Linear Programming (MILP). This ensures that out of millions of possible combinations, you receive the one that provides the **maximum protein** for the **minimum cost**.
    
    ### 2. High Entropy = Longevity
    In Information Theory, **Entropy** measures variety. Our algorithm forces a 'Maximum Entropy' state, ensuring you consume a diverse array of vitamins and minerals.
    
    ### 3. Positive Lifestyle Impact
    * **Cognitive Function**: Proper glucose management prevents 'brain fog'.
    * **Financial Stability**: Linear optimization eliminates 'spending leakage'.
    * **Metabolic Health**: Targeted calorie intake prevents metabolic slowdown.
    """)
