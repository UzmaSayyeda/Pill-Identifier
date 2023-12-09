let response = '/data'

d3.json(response).then((data) => {

    let drugNames = data.map((entry) => entry.drug_name)
    // console.log(drugNames)

    let x = drugNames.slice(0,10).reverse()
   // console.log(x)

    let medicalCond = data.map((entry) => entry.medical_condition)
    // console.log(medicalCond)

    let y = medicalCond.slice(0,10).reverse()
    // console.log(y)

    let barData = [

        {   x : x,
            y : y,
            type : "bar",
            
        
        }
    ];

    let barLayout = {
        title : "Top ten drugs"

    };

    Plotly.newPlot("myChart", barData, barLayout)
})


// document.addEventListener("DOMContentLoaded", function () {
//     fetch("/data")
//       .then((response) => response.json())
//       .then((data) => {
//         // Extract relevant data for the chart
//         const labels = data.map((entry) => entry.drug_name);
//         const sideEffects = data.map((entry) => entry.side_effects);
  
//         // Create a bar chart
//         var ctx = document.getElementById("myChart").getContext("2d");
//         var myChart = new Chart(ctx, {
//           type: "bar",
//           data: {
//             labels: labels,
//             datasets: [
//               {
//                 label: "Side Effects",
//                 data: sideEffects,
//                 backgroundColor: "rgba(75, 192, 192, 0.2)",
//                 borderColor: "rgba(75, 192, 192, 1)",
//                 borderWidth: 1,
//               },
//             ],
//           },
//           options: {
//             scales: {
//               y: {
//                 beginAtZero: true,
//               },
//             },
//           },
//         });
//       })
//       .catch((error) => {
//         console.error("Error fetching data:", error);
//       });
//   });