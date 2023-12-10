

// Get the prediction value and store it


// List of pills with descriptions and possible side effects
var pills = {
'Amoxicillin 500 mg': {
    Description: 'Used to treat bacterial infections, such as chest infections and dental abscesses.',
    PossibleEffects: 'Possible side effects include nausea and diarrhea.'
},
'Apixaban 2.5 mg': {
    Description: 'Used to treat or prevent deep venous thrombosis and pulmonary embolism.',
    PossibleEffects: 'May cause longer bleeding time and increased bruising.'
},
'Aprepitant 80 mg': {
    Description: 'Prevents nausea and vomiting after cancer chemotherapy.',
    PossibleEffects: 'Common side effects include headache, fatigue, and constipation.'
},
'Atomoxetine 25 mg': {
    Description: 'Works in the brain to increase attention and decrease restlessness in hyperactive individuals.',
    PossibleEffects: 'Common side effects include upset stomach, nausea, and trouble sleeping.'
},
'Calcitriol 0.00025': {
    Description: 'Treats low calcium levels caused by kidney disease and regulates parathyroid levels.',
    PossibleEffects: 'May cause restlessness, difficulty thinking clearly, loss of appetite, and nausea.'
    },
'Prasugrel 10 MG': {
    Description: 'Prevents platelets from forming clots that may cause heart attacks or strokes.',
    PossibleEffects: 'Common side effects include blurred vision, dizziness, and headache.'
},
'Ramipril 5 MG': {
    Description: 'Lowers blood pressure by relaxing blood vessels.',
    PossibleEffects: 'May cause sudden swelling of face, arms, legs, lips, and throat.'
},
'Saxagliptin 5 MG': {
    Description: 'Used to treat high blood sugar levels in patients with type 2 diabetes.',
    PossibleEffects: 'May cause pancreatitis and other digestive issues.'
},
'Sitagliptin 50 MG': {
    Description: 'Helps control blood sugar levels by increasing insulin release and reducing liver sugar production.',
    PossibleEffects: 'May cause symptoms like fever, nausea, and loss of appetite.'
},
'Tadalafil 5 MG': {
    Description: 'Used to treat erectile dysfunction in men.',
    PossibleEffects: 'May cause difficulty urinating and painful urination.'
},
'carvedilol 3.125': {
    Description: 'A beta blocker that slows down the heart rate and lowers blood pressure.',
    PossibleEffects: 'Common side effects include dizziness, lightheadedness, and diarrhea.'
},
'celecoxib 200': {
    Description: 'A nonsteroidal anti-inflammatory drug (NSAID) used to treat pain and inflammation.',
    PossibleEffects: 'May cause stomach pain, heartburn, vomiting, and black stools.'
},
'duloxetine 30': {
    Description: 'An antidepressant that increases serotonin and noradrenaline levels in the brain.',
    PossibleEffects: 'May cause angle-closure glaucoma and other side effects.'
},
'eltrombopag 25': {
    Description: 'Used to increase platelet count in people with certain conditions.',
    PossibleEffects: 'May cause serious liver problems and other side effects.'
},
'metformin_500': {
    Description: 'Lowers blood sugar levels in diabetes patients.',
    PossibleEffects: 'May cause general unwell feeling, fast or shallow breathing, and yellowing of eyes or skin.'
},
'montelukast-10': {
    Description: 'Used to prevent wheezing and breathing difficulties caused by asthma.',
    PossibleEffects: 'May cause difficulty breathing, rash, and flu-like symptoms.'
},
'mycophenolate-250': {
    Description: 'Used to help prevent organ rejection after transplantation.',
    PossibleEffects: 'May cause constipation, nausea, headache, and other side effects.'
},
'omeprazole_40': {
    Description: 'Reduces stomach acid and treats indigestion, heartburn, and acid reflux.',
    PossibleEffects: 'Common side effect is headache, and it may cause stomach-related issues.'
},
'oseltamivir-45': {
    Description: 'Used to treat influenza infections.',
    PossibleEffects: 'May cause confusion, tremors, and unusual behavior.'
},
'pantaprazole-40': {
    Description: 'Reduces stomach acid and treats heartburn and gastroesophageal reflux disease (GERD).',
    PossibleEffects: 'May cause infectious diarrhea, kidney problems, and skin reactions.'
},
'pitavastatin_1': {
    Description: 'A statin that lowers cholesterol levels in the body.',
    PossibleEffects: 'May cause dark urine, diarrhea, fever, and muscle problems.'
},
'prednisone_5': {
    Description: 'Treats various conditions by reducing inflammation and suppressing the immune system.',
    PossibleEffects: 'May cause nausea, vomiting, heartburn, and trouble sleeping.'
},
'sertraline_25': {
   Description: 'An antidepressant that increases serotonin levels in the brain.',
   PossibleEffects: 'Common side effects include nausea, changes in sleep patterns, and diarrhea.'
},
};

console.log("Prediction:", prediction);

// Check if the prediction is in the list
if (pills.hasOwnProperty(prediction)) {
    var pillInfo = pills[prediction];
    var pillHTML = "<div style='margin-top: 220px; border: 2px solid #427D9D; border-radius: 12px; box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19); padding: 10px; width: fit-content; margin: auto; background-color: #427D9D; color: black; font-weight: bold; opacity: 0.8;'>";
    pillHTML += "<p>Pill Name: " + prediction + "</p>";
    pillHTML += "<p>Uses: " + pillInfo.Description + "</p>";
    pillHTML += "<p>Possible Side Effects: " + pillInfo.PossibleEffects + "</p>";
    pillHTML += "</div>";

    // Replace the content of predictionElement with the pillHTML
    document.getElementById("predictionElement").innerHTML = pillHTML;
} else {
    // Display a message if the prediction is not in the list
    document.write("<p>No information is available for this pill.</p>");
}