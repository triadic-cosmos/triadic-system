# Generate stories with model
from engine.triadic_writer import TriadicWriter

import time

# Generation main using keywords, prompt and beam search
start = time.perf_counter()

print("Generating stories...")

model = "odyssey"
prefix = "200k"
number_lines = 20
number_stories = 10
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

writer: TriadicWriter = TriadicWriter(model, prefix, number_lines)
writer.write(number_stories, ODYSSEY_PROMPT, keywords, beam_search)

print(f"Time: {time.perf_counter() - start:.1f} s")
