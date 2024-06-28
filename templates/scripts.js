document.getElementById('loanForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const formData = new FormData(this);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/predictdata', true);
    xhr.onload = function() {
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);
            const predictionElement = document.getElementById('predictionResult');
            predictionElement.textContent = `THE prediction is ${response.prediction}`;
            predictionElement.style.display = 'block';
        }
    };
    xhr.send(formData);
});
