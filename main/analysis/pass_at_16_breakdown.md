# Pass@16 Analysis: Qwen2.5-1.5B-Instruct on GSM8K

**Model**: Qwen/Qwen2.5-1.5B-Instruct
**Samples**: 100 GSM8K test questions, 16 solutions each at temp=0.7
**Estimated pass@1**: 67.7% | **Pass@16**: 95%

---

## Confirmation: This is the 1.5B model

- 0.5B had 45% pass@1 (greedy baseline)
- 3B was too strong (nearly all 16/16 in partial results)
- This data shows 67.7% pass@1 and a healthy spread of difficulty -- consistent with 1.5B

---

## Difficulty Distribution

```
 0/16  (hopeless):     5 problems   |#####
 1-2   (very hard):    5 problems   |#####
 3-5   (hard):         9 problems   |#########
 6-10  (medium):      19 problems   |###################
 11-14 (nearly there): 30 problems  |##############################
 15-16 (easy):        32 problems   |################################
```

---

## HOPELESS: 0/16 -- Model NEVER gets these right

| # | Question (summary) | Gold | What goes wrong |
|---|---|---|---|
| 1 | 4 people raising money for carnival (Kim, Alexandra, Maryam, Sarah) | 2280 | Forgets to include Sarah's $300 in the total. Adds only 3 of 4 people. |
| 2 | Tom's ship: sails 1-4 PM at 10mph, returns at 6mph | 5 | Confuses "3 hours" (time) with distance. Does 3/10 instead of 3*10=30 miles first. |
| 3 | Elise re-writing alphabet (full twice, half once, then re-writes everything) | 130 | Misinterprets "re-writes everything she has already written" -- loses track of the doubling. |
| 4 | Mike replacing 600 movies (1/3 series at $6, 40% older at $5, rest at $10) | 4400 | Fails to handle the nested structure: 1/3 series, then 40% of REMAINING, then rest. |
| 5 | Abraham sells land: half for $50, then 1/4 for $30, rest at $3/sqm | 170 | Misreads "1/4 of his land" -- is it 1/4 of original or 1/4 of remaining? |

**Pattern**: Multi-step problems with **nested references** or **ambiguous pronouns** ("his land", "everything she wrote"). The model loses track of what quantity refers to what.

---

## VERY HARD: 1-2 out of 16

| # | Question (summary) | Gold | n_correct | What goes wrong |
|---|---|---|---|---|
| 1 | Ten stalls with 20 cows, Sylas buys 40 more, divides into "twenty stalls" | 192 | 1/16 | Confused by "ten stalls" vs "twenty stalls" in the problem text |
| 2 | Aaron vs Vanessa relay race (speeds and distances) | 64 | 1/16 | Mixes up the relationship between speed, distance, and time |
| 3 | Gomer ate "5 less than 23" scoops, potato-to-scoop conversion | 27 | 1/16 | "3 less than 6 potatoes to make 1 less than 3 scoops" -- nested subtractions |
| 4 | Toys in a room: doll costs "as much as 3 action figures" | 50 | 2/16 | Misreads "the doll cost as much as 3 action figures" price relationship |
| 5 | Frederick making popsicle sticks (cost optimization) | 1600 | 2/16 | Optimization problem -- needs to compare cost-per-stick and buy cheapest |

**Pattern**: Problems with **confusing phrasing** ("5 less than 23", "1 less than 3 scoops") or **implicit optimization** (buy the cheapest option).

---

## HARD: 3-5 out of 16

| # | Question (summary) | Gold | n_correct | Key failure |
|---|---|---|---|---|
| 1 | Twenty dozen cups vs half dozen plates at $6000 each | 145 | 3/16 | Misreads "$6000 each" as per-dozen instead of per-plate |
| 2 | Jen's 3 fish, $1/day food, month of May | 93 | 3/16 | Uses 30 days for May instead of 31 |
| 3 | Doubtfire sisters: Patchy had "thrice the adopted kittens" | 40 | 4/16 | "Thrice the number of adopted kittens" -- wrong referent (7 vs 12) |
| 4 | Manolo: 5 lollipops + 4 candies = $3.20 total | 7 | 4/16 | Misreads "$3.20" as the cost of lollipops only, not total |
| 5 | Hannah smashes 1/4 of students' and 3/4 of teachers' car windows | 112 | 4/16 | Confuses "1/4 of cars" vs "1/4 of windows" |
| 6 | Josh's car shop: open every day except Sunday AND Wednesday | 120 | 5/16 | Forgets Wednesday is also closed (counts 6 days instead of 5) |
| 7 | Elvis saving twice as much in second half of April | 50 | 5/16 | Misinterprets "first half" and "second half" of month (15 days each) |
| 8 | Mark lost 10 lbs/month for 3 months, final weight 70 | 100 | 5/16 | Subtracts instead of adds to find initial weight |
| 9 | Elaine's Pokemon cards over 3 months with multipliers | 320 | 5/16 | Loses track of "combined" across months |

**Pattern**: **Misreading modifiers** ("each", "total", "per"), **factual knowledge** (days in May), and **directional errors** (subtract vs add to find original).

---

## MEDIUM: 6-10 out of 16

| # | Question (summary) | Gold | n_correct | Key failure |
|---|---|---|---|---|
| 1 | Hunter counting cars (50 initial + 20 more, half leave) | 35 | 6/16 | Confused about "half had gone" -- half of what? |
| 2 | Susan earns $5/10min, works 8-11am with 30min pause | 75 | 6/16 | Miscalculates working time with the pause |
| 3 | Brian's friend Bobby: "5 fewer than 3 times" Brian's games | 40 | 6/16 | Applies formula to wrong version of Brian's count |
| 4 | Elliott's steps: walks, jog, 2000 left | 2000 | 6/16 | Misinterprets "only had 2000 steps LEFT to take" |
| 5 | Emma's vlogs: 18+21+15, need 72 total | 18 | 7/16 | Some completions misread the question |
| 6 | Mass drills: 8 in a row, 7 rows, 5 schools | 280 | 8/16 | Forgets to multiply by 5 schools |
| 7 | Commission: 30% on first $1000, 10% on rest | 450 | 8/16 | Misreads "additional 10%" as the ONLY rate above $1000 |
| 8 | Family ages: 2 brothers, 3 sisters (all 16), one is 12 = half of older | 84 | 9/16 | Misreads who is "half the age of" whom |
| 9 | Lee vs Gerald running: "two seconds faster", 10% improvement | 36 | 9/16 | Confuses "faster" = lower time, 10% improvement on speed vs time |
| 10 | Jim: 2hr TV + half reading, 3x/week, 4 weeks | 36 | 9/16 | Confuses "3 times a week" with "3 days" or "4 days" |
| 11 | Birthday candles: 12yo and "4 years younger", packs of 5 | 12 | 10/16 | Some completions think 1 candle per person, not per year |
| 12 | Randy cookies: multiple operations on 3 flavors | 23 | 10/16 | Loses track during sequential eat/give/bake operations |

**Pattern**: **Sequential multi-step operations**, **"faster/slower" directional confusion**, **misreading relative references** ("3 times a week" vs "3 days").

---

## FAILURE MODE SUMMARY

| Failure Category | Count | Description |
|---|---|---|
| **Misread referent/modifier** | ~25 | "thrice the adopted kittens" → which number does "thrice" apply to? |
| **Entity tracking** | ~10 | Forgetting a person, item, or constraint mid-problem |
| **Directional error** | ~8 | Subtract instead of add, "faster" means less time not more |
| **Factual knowledge** | ~3 | Days in May (31), alphabet has 26 letters |
| **Nested structure** | ~5 | "40% of the remaining" after already taking 1/3 |
| **Optimization/multi-part** | ~4 | Comparing options, handling conditional pricing |

---

## POTENTIAL TARGETED TRAINING SETS

Based on the failure patterns above, here are focused datasets you could build:

### Dataset 1: "Who Has What?" (Entity Tracking)
Problems with 3+ people/items where the model must track each one separately.
Targets: idx 1309, 54, 178, 334, 696, 1287

### Dataset 2: "Careful Reading" (Modifier/Referent)
Problems where a phrase like "twice as many as X" or "$6000 each" must be parsed precisely.
Targets: idx 209, 689, 727, 940, 163, 552

### Dataset 3: "Step-by-Step Sequential" (Multi-operation)
Problems requiring 4+ operations in sequence without losing track.
Targets: idx 1301, 407, 864, 65, 206, 538, 821

### Dataset 4: "Tricky Phrasing" (Nested/Ambiguous)
Problems with "5 less than 23" or "1/4 of his remaining land" type constructions.
Targets: idx 1140, 1161, 546, 777, 541, 135
