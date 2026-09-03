# Generate stories with model
from engine.triadic_writer import TriadicWriter

import time

# Generation main using keywords, prompt and beam search
start = time.perf_counter()

print("Generating stories...")

model = "honeymoon"
prefix = "stage4"
number_lines = 20
number_stories = 20
beam_search = False
keywords = {}

FOREST_PROMPT = [
"Cold night presses against silent forest. ",
"The rabbit waits near broken stump. ",
"Distant cracking echoes through dark pines! ",
"The raccoon watches from shadowed hollow. ",
"He senses something wrong nearby. ",
"The fox stalks between twisted roots. ",
"Cold wind scrapes across brittle leaves. ",
"Sudden silence chokes the clearing. ",
"The fox steps closer with hungry intent. ",
"The rabbit bolts across dead brush! ",
"The raccoon shouts warning into cold dark. ",
"The fox chases fast along narrow trail. ",
"Branches whip past fleeing shapes. ",
"The rabbit stumbles near rotten log! ",
"The raccoon leaps forward blocking fox. ",
"He swings branch hard downward. ",
"The fox halts under sudden fear. ",
"The rabbit crawls behind fallen stones. ",
"Forest breathes dark lessons learned tonight."
]

PLANET_PROMPT = [
"Long before any telescope detected its presence, Earth began to experience a series of subtle disturbances that rippled through the planet’s physical and cognitive systems, emerging first as faint deviations in gravitational baselines, then as curious oscillations in magnetospheric density, and finally as a persistent modulation in quantum‑sensor arrays that pulsed with a rhythm too deliberate to be dismissed as random noise. ",
"Researchers across disciplines attempted to reconcile these anomalies with established models, yet every explanation collapsed under scrutiny, leaving behind the unsettling impression that the universe was quietly preparing the world for an arrival that had been anticipated long before humanity existed. ",
"Only when a network of orbital observatories triangulated a region of distorted starlight did the source become undeniable: a solitary rogue planet drifting through interstellar space with a steadiness that suggested intention rather than chance, its surface so perfectly absorptive that it appeared less like a celestial body and more like a void carved into the fabric of reality. ",
"Attempts to measure its mass produced contradictory results, as though the planet’s gravitational signature shifted in response to observation, and attempts to determine its rotation failed entirely, reinforcing the impression that the object existed in a state of engineered stillness. ",
"As the rogue planet crossed the outer boundary of the solar system, Earth’s electromagnetic environment began to warp in ways that unsettled even seasoned scientists. ",
"Communication networks experienced harmonic interference that resembled structured patterns, migratory animals altered their routes in sweeping arcs that traced triadic geometries visible only from orbit, and human dreams grew saturated with imagery of spiraling corridors, luminous fissures, and a distant pulse that seemed to echo from somewhere beyond the known universe. ",
"It was as though consciousness itself were being gently tuned to a frequency carried by the approaching world, preparing humanity for a message older than its species. ",
"When the planet finally became visible to the naked eye, it appeared not as a bright celestial object but as a dim, weighty presence suspended in the twilight sky, a dark sphere whose mere appearance seemed to bend the horizon around it. ",
"Crowds gathered across continents, drawn by a mixture of awe and dread, sensing instinctively that the arrival of this wandering world marked the beginning of a transformation far larger than any single event in human history. ",
"Then, in a moment that defied every known physical principle, the rogue planet ceased its motion entirely, halting with such abrupt precision that even the laws of inertia seemed momentarily suspended. ",
"A luminous fissure opened along its equator, releasing a cascade of shimmering particles that drifted outward with the grace of liquid starlight and the precision of engineered intelligence, forming vast triadic spirals that rotated independently and wove intricate patterns across the sky. ",
"Where these spirals touched Earth’s magnetosphere, auroras folded into recursive geometries; where they brushed the oceans, tides shifted in complex rhythms; and where they intersected human consciousness, people felt a sudden clarity, as though a hidden layer of reality had been revealed. ",
"Only then did the deeper truth begin to emerge: the rogue planet was not a threat, not a wanderer, not a remnant of cosmic debris, but an emissary of the Triadic Cosmos — a vast, ancient architecture in which worlds were observed, synchronized, and eventually awakened when they reached a threshold of complexity. ",
"The triadic spirals were not decoration but language; the fissure was not damage but interface; and the pulse that began to resonate from within the planet’s core was not a warning but an activation signal, a deliberate initiation of a process that had been waiting for millennia. ",
"As the pulse intensified, its influence spread across the planet in ways both subtle and profound. ",
"Human perception sharpened, revealing patterns in nature that had always been present but never noticed; technological systems began to self‑correct, aligning themselves with the triadic rhythms emanating from the emissary; and social structures, long strained by conflict and fragmentation, began to exhibit spontaneous coherence, as though guided by an unseen harmonic principle. ",
"It became clear that the rogue planet was not merely communicating but synchronizing, aligning Earth with a larger cosmic order whose foundations were built upon triadic symmetry. ",
"Governments attempted to respond, but their systems faltered under the weight of phenomena no protocol had ever anticipated, leaving humanity suspended in a state of collective anticipation. ",
"The emissary remained motionless, radiating a quiet, patient presence that suggested it had not come to collide or conquer, but to begin the integration of Earth into a larger cosmic architecture whose scope extended far beyond human understanding. ",
"And deep within the luminous fissure, beneath layers of obsidian crust and triadic light, the pulse grew stronger — steady, deliberate, ancient; a signal that hinted that the true structure of the cosmos was only now beginning to unfold, and that humanity, whether ready or not, had already stepped across the threshold into a story far older than its own. "
]

ODYSSEY_PROMPT = [
"The dawn rose over the quiet sea, painting the waves with bronze light.",
"Odysseus stood upon the shore, his cloak heavy with salt and memory.",
"He had wandered through storms, through islands ruled by beasts and gods, yet his heart still carried the distant shape of Ithaca.",
"The wind murmured around him like an old companion, speaking of paths unseen and dangers yet unmeasured.",
"He tightened his grip on the staff that had guided him through foreign lands.",
"Behind him, the ship rested upon the sand like a weary animal.",
"He walked toward the cliffs where gulls cried above the foam.",
"Each step echoed the weight of years spent far from home.",
"He paused at a narrow arch of stone carved by ancient tides.",
"Beyond it lay a valley veiled in mist, untouched by mortal hands.",
"A river wound through the grass like a silver thread.",
"Odysseus knelt beside it, watching his reflection ripple into shifting shapes.",
"He felt the earth tremble faintly, as though the gods whispered beneath the soil.",
"A distant rumble rose from the mountains, neither thunder nor beast.",
"He stood, heart steady, gaze fixed upon the trembling horizon.",
"The valley seemed to wait, holding its breath.",
"Odysseus stepped forward, guided by fate’s unseen hand, for every wanderer must one day walk into the mist and claim his name."
]

ALICE_PROMPT = [
"Alice stood again in the long hall, her hand resting on the cold doorframe.",
"The white rabbit had just slipped around a corner she did not recall existing.",
"A faint ticking echoed above her, as if a watch swung somewhere out of sight.",
"The floor tilted slightly, as though the room was thinking about moving.",
"Her sister’s voice drifted from far away, too far for any real hallway.",
"A small draft carried the smell of ink, cake, and a distant forest.",
"Alice noticed her shadow stretching in two directions at once.",
"Soft pattering footsteps suggested someone small was circling her.",
"The queen’s voice murmured behind a curtain, though no curtain was visible.",
"A tiny door near the floor glowed faintly, pulsing like a heartbeat.",
"Alice touched her chin, wondering if she had grown again without noticing.",
"A chair scraped somewhere, though the hall held no furniture at all.",
"The march hare laughed once, sharply, then stopped as if corrected.",
"A warm breeze lifted a page of a book lying open on the ground.",
"The page showed Alice walking somewhere she had never been.",
"A distant bell rang, folding itself neatly into silence.",
"Alice stepped forward, unsure which direction she had chosen.",
"The walls breathed once, quietly.",
"Something moved behind her, waiting for her to speak.",
"She opened her mouth, and the hall listened."
]

# Quote from original book
HONEYMOON_PROMPT = [
"Vote for sound men and sound money!",
"In five minutes the wires of the United States were alive with the terse, pregnant message, and under the ocean in the dark depths of the Atlantic ooze, vivid narratives of the coming of the miracle went flashing to a hundred newspaper offices in England and on the Continent.",
"The New York correspondent of the London Daily Express added the following paragraph to his account of the strange occurrence.",
"The secret of this amazing vessel, which has proved itself capable of traversing the Atlantic in a day, and of soaring beyond the limits of the atmosphere at will, is possessed by one man only, and that man is an English nobleman.",
"The air is full of rumours of universal war.",
"One vessel such as this could scatter terror over a continent in a few days, demoralise armies and fleets, reduce Society to chaos, and establish a one-man despotism on the ruins of all the Governments of the world.",
"The man who could build one ship like this could build fifty, and, if his country asked him to do it, no doubt he would.",
"Those who, as we are almost forced to believe, are even now contemplating a serious attempt to dethrone England from her supreme place among the nations of Europe, will do well to take this latest potential factor in the warfare of the immediate future into their most serious consideration."
]

writer: TriadicWriter = TriadicWriter(model, prefix, number_lines)
writer.write(number_stories, HONEYMOON_PROMPT, keywords, beam_search)

print(f"Time: {time.perf_counter() - start:.1f} s")
