let response = '/data'


d3.json(response).then((data) => {

    // get rx_otc from /data
    let rxOtc = data.map((entry) => entry.rx_otc)

    let x = rxOtc

    // plot histogram with rx_otc
    let DataRx = [

        {
            x : x,        
            type : "histogram"
         
        }
    ];

    // set up layout
    let LayoutRx = {
        title : "Rx OTC"

    };
    // display 
    Plotly.newPlot("view-2", DataRx, LayoutRx)
})

d3.json(response).then((data) => {

    // get medical_condition and map 
    let medicalCond = data.map((entry) => entry.medical_condition)

    let  x2 = medicalCond

    // plot the medical condtition histogram
    let DataMed = [

        {
            x : x2,            
            type : "histogram"
         
        }
    ];

    // define the layout
    let LayoutMed = {
        title : "Medical Condition"

    };
    
    // display using html tag id, bar data and layout
    Plotly.newPlot("view-3", DataMed, LayoutMed)
})
 