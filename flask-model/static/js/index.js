let response = "/data"

d3.json(response).then(function(data){
    console.log(data)
})

document.addEventListener("DOMContentLoaded", function () {
    fetch("/data")
      .then((response) => response.json())
      .then((data) => {
        // Extract relevant data for the chart
        const labels = data.map((entry) => entry.drug_name);
        const sideEffects = data.map((entry) => entry.side_effects);
  
        // Create a bar chart
        var ctx = document.getElementById("myChart").getContext("2d");
        var myChart = new Chart(ctx, {
          type: "bar",
          data: {
            labels: labels,
            datasets: [
              {
                label: "Side Effects",
                data: sideEffects,
                backgroundColor: "rgba(75, 192, 192, 0.2)",
                borderColor: "rgba(75, 192, 192, 1)",
                borderWidth: 1,
              },
            ],
          },
          options: {
            scales: {
              y: {
                beginAtZero: true,
              },
            },
          },
        });
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
  });