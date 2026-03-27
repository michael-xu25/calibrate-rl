"""
Build Entity-Tracking Dataset from GSM8K.

Filters GSM8K training set to problems with 3+ named entities
(people/items the model must track individually).

Adds an 'entities' column with pre-extracted entity names for each problem.

Usage:
    python src/build_entity_dataset.py

Output:
    data/entity_tracking_dataset/  (HuggingFace arrow format)
"""

import re
from datasets import load_dataset

# ── Non-name words to filter out ────────────────────────────────────────────
# These are capitalized words that appear in GSM8K questions but are NOT
# entity names (people, pets, stores, etc.)

DAYS = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
}

MONTHS = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}

# Common sentence starters, pronouns, and other false positives
COMMON_NON_NAMES = {
    # Pronouns / determiners / conjunctions
    "If", "How", "What", "When", "Where", "Who", "Why", "Which",
    "The", "He", "She", "They", "His", "Her", "It", "Its", "Their",
    "Each", "Every", "Some", "All", "One", "Two", "Three", "Four",
    "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "This", "That", "These", "Those", "There", "Here",
    "After", "Before", "During", "Since", "While", "Until",
    "In", "On", "At", "For", "From", "By", "To", "Of", "With",
    "But", "And", "Or", "So", "Yet", "Then", "Than", "Also",
    "An", "A", "As", "Be", "Do", "Is", "Was", "Are", "Were",
    "Has", "Had", "Have", "Can", "Could", "Would", "Should",
    "Not", "No", "Yes", "Both", "Many", "Most", "Only",

    # Math/problem words that get capitalized
    "Step", "Answer", "Total", "Calculate", "Find", "Determine",
    "Given", "Let", "Solve", "Note", "Problem", "Solution",
    "Half", "Twice", "Double", "Triple", "Quarter",

    # Common non-person proper nouns in GSM8K
    "UV", "Art", "TV", "DVD", "GPS", "ATM", "GPA", "SAT", "ACT",
    "NFL", "NBA", "MLB", "USA", "US", "UK", "PM", "AM",

    # Places and geography
    "North", "South", "East", "West", "Central",
    "City", "Town", "Street", "Avenue", "Road", "Park", "Drive",
    "Lake", "River", "Mountain", "Island", "Valley", "Beach", "Bay",
    "York", "Angeles", "Francisco", "Diego",
    "America", "American", "European", "African", "Asian",
    "German", "French", "Chinese", "Japanese", "Italian", "Spanish",
    "English", "British", "Mexican", "Canadian",

    # Organizations / landmarks / brands
    "Facebook", "Instagram", "Twitter", "YouTube", "Amazon", "Google",
    "Metropolitan", "Museum", "Library", "University", "College",
    "Hospital", "Church", "Airport", "Station",
    "National", "International", "Local", "Royal",
    "Walmart", "Target", "Costco",

    # Common nouns that may appear capitalized at sentence start
    "Student", "Students", "Teacher", "Teachers", "Farmer", "Class",
    "Children", "People", "Workers", "Company", "Store", "School",
    "Family", "Friends", "Team", "Group", "Member", "Members",
    "Boy", "Girl", "Boys", "Girls", "Man", "Woman", "Men", "Women",
    "Brother", "Sister", "Mother", "Father", "Uncle", "Aunt",
    "Son", "Daughter", "Husband", "Wife", "Baby",
    "Mom", "Dad", "Grandma", "Grandpa", "Grandad",
    "Grandmother", "Grandfather", "Cousin", "Neighbor",
    "Day", "Week", "Month", "Year", "Hour", "Minute",
    "Morning", "Afternoon", "Evening", "Night",

    # Genres / categories / game names appearing in GSM8K
    "Western", "Eastern", "Northern", "Southern",
    "Lego", "Legos", "Scrabble", "Pokemon",
    "Junior", "Senior", "Little", "Big", "Great",
    "Christmas", "Easter", "Halloween", "Thanksgiving",

    # Ordinals and misc
    "First", "Second", "Third", "Fourth", "Fifth", "Last", "Next",
    "New", "Old", "Once", "Twice", "Per", "Plus",
    "Currently", "Originally", "Recently", "Finally",
    "However", "Therefore", "Meanwhile", "Although",

    # Titles (the name usually follows)
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sir",
}

# Merge all exclusions
ALL_EXCLUSIONS = DAYS | MONTHS | COMMON_NON_NAMES


def _strip_possessive(name: str) -> str:
    """Strip possessive forms: Boyd's -> Boyd, Boyds -> Boyd, Amilia's -> Amilia.

    Handles both apostrophe possessives and bare-s possessives.
    Avoids stripping 's' from names that naturally end in 's' (James, Charles)
    by only stripping if the base form is 3+ characters.
    """
    # "Boyd's" or "Amilia's" -> strip "'s"
    if name.endswith("'s") and len(name) > 3:
        return name[:-2]
    # "Boyds" (no apostrophe) -> strip "s" if base is 3+ chars
    # But don't strip from names like "James", "Charles" -- we check
    # if it ends in a consonant + 's' pattern typical of possessives
    if (name.endswith("s")
            and not name.endswith("ss")  # "Ross", "Jess"
            and len(name) > 3):
        return name[:-1]
    return name


def extract_entity_names(question: str) -> list[str]:
    """Extract entity names (proper nouns) from a GSM8K question.

    Strategy:
    1. Split text into sentences
    2. For each sentence, find capitalized words (2+ chars, alphabetic)
    3. Strip possessives (Boyd's -> Boyd, Boyds -> Boyd)
    4. Filter out known non-names (places, brands, common words)
    5. Filter: both the raw form AND base form must pass exclusion check
    6. Deduplicate (case-insensitive, also dedup base forms)

    First word of each sentence is included only if it's clearly a name
    (not in the exclusion list).
    """
    # Split into sentences on . ? ! and also handle newlines
    sentences = re.split(r'[.?!\n]+', question)

    candidates = []

    for sent in sentences:
        words = sent.strip().split()
        if not words:
            continue

        for idx, word in enumerate(words):
            clean = re.sub(r"[^a-zA-Z']", "", word)
            if not clean or len(clean) < 2:
                continue
            # Must start with uppercase
            if not clean[0].isupper():
                continue
            # Must not be ALL uppercase (acronyms handled in exclusions)
            if clean.isupper() and len(clean) > 2:
                continue

            # Strip possessive to get base name
            base = _strip_possessive(clean)

            # Both raw and base form must pass exclusion filter
            if base in ALL_EXCLUSIONS or clean in ALL_EXCLUSIONS:
                continue
            if len(base) < 2:
                continue

            candidates.append(base)

    # Deduplicate case-insensitively, preserving first occurrence
    seen = set()
    unique = []
    for name in candidates:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)

    return unique


def main():
    print("=" * 70)
    print("  Building Entity-Tracking Dataset from GSM8K")
    print("=" * 70)

    # Load GSM8K train split
    print("\n>>> Loading GSM8K train split ...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    print(f"    Total problems: {len(dataset)}")

    # Extract entities for each problem
    print("\n>>> Extracting entity names ...")

    def add_entities(example):
        example["entities"] = extract_entity_names(example["question"])
        return example

    dataset = dataset.map(add_entities)

    # Stats before filtering
    entity_counts = [len(e) for e in dataset["entities"]]
    print(f"\n>>> Entity count distribution (all {len(dataset)} problems):")
    for n in range(6):
        count = sum(1 for c in entity_counts if c == n)
        bar = "#" * (count // 20)
        print(f"    {n} entities: {count:>5} problems  {bar}")
    count_6plus = sum(1 for c in entity_counts if c >= 6)
    print(f"   6+ entities: {count_6plus:>5} problems")

    # Filter for 3+ entities
    min_entities = 3
    filtered = dataset.filter(lambda x: len(x["entities"]) >= min_entities)
    print(f"\n>>> Filtered to {min_entities}+ entities: {len(filtered)} problems")

    # If too few, fall back to 2+
    if len(filtered) < 1000:
        print(f"    Only {len(filtered)} problems with {min_entities}+ entities.")
        min_entities = 2
        filtered = dataset.filter(lambda x: len(x["entities"]) >= min_entities)
        print(f"    Relaxed to {min_entities}+ entities: {len(filtered)} problems")

    # Show sample problems
    print(f"\n>>> Sample problems ({min(5, len(filtered))} examples):")
    print("-" * 70)
    for i in range(min(5, len(filtered))):
        ex = filtered[i]
        q = ex["question"][:200] + ("..." if len(ex["question"]) > 200 else "")
        entities = ex["entities"]
        print(f"  [{i+1}] Entities: {entities}")
        print(f"      Q: {q}")
        print()

    # Save the filtered dataset
    output_path = "data/entity_tracking_dataset"
    print(f">>> Saving to {output_path} ...")
    filtered.save_to_disk(output_path)
    print(f"    Saved {len(filtered)} problems with {min_entities}+ entities.")

    # Final summary
    filtered_entity_counts = [len(e) for e in filtered["entities"]]
    avg_entities = sum(filtered_entity_counts) / len(filtered_entity_counts)
    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"  Total problems:    {len(filtered)}")
    print(f"  Avg entities/prob: {avg_entities:.1f}")
    print(f"  Min entities:      {min(filtered_entity_counts)}")
    print(f"  Max entities:      {max(filtered_entity_counts)}")
    print(f"  Output:            {output_path}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
