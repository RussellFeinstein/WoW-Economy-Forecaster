"""
Crafting recipe data layer.

Modules:
  blizzard_recipe_client  — wraps BlizzardClient static-API profession/recipe calls
  recipe_seeder           — orchestrates fetch -> normalize -> upsert into DB
  recipe_repo             — data access (read-only queries over recipes/reagents)
  margin_calculator       — computes daily crafting_margin_snapshots from price data
"""
