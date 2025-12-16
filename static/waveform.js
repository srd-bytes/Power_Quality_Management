function updateWaveform() {
    fetch('/waveform_data')
        .then(response => response.json())
        .then(data => {
            const labelColors = {
                normal: 'red',
                sag: 'blue',
                swell: 'green',
                spike: 'orange'
            };

            const trace = {
                x: data.x,
                y: data.y,
                type: 'scatter',
                mode: 'lines',
                line: { color: labelColors[data.label], width: 2 }
            };

            Plotly.newPlot('plot', [trace], {
                title: `Waveform - ${data.label.toUpperCase()}`,
                margin: { t: 30 }
            });
        });
}

setInterval(updateWaveform, 1000);
updateWaveform();
