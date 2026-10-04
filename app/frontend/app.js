const question = document.querySelector('#question');
const columns = document.querySelector('#columns');
const generateButton = document.querySelector('#generate');
const status = document.querySelector('#status');
const greedy = document.querySelector('#greedy');
const beam = document.querySelector('#beam');
generateButton.addEventListener('click', async () => {
    status.textContent = 'Generating SQL...';
    generateButton.disabled = true;
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question.value,
                columns: columns.value
                    .split(/[\n,]/)
                    .map(column => column.trim())
                    .filter(Boolean)
            })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'The request could not be completed.');
        greedy.textContent = data.sql || 'Could not parse the model output.';
        beam.textContent = data.query
            ? JSON.stringify(data.query, null, 2)
            : 'Could not parse the model output.';
        status.textContent = 'Complete.';
    } catch (error) {
        status.textContent = error.message;
    } finally {
        generateButton.disabled = false;
    }
});