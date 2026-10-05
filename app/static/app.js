const question = document.querySelector('#question');
const columns = document.querySelector('#columns');
const generateButton = document.querySelector('#generate');
const status = document.querySelector('#status');
const greedySql = document.querySelector('#greedy-sql');
const greedyQuery = document.querySelector('#greedy-query');
const beamSql = document.querySelector('#beam-sql');
const beamQuery = document.querySelector('#beam-query');

const elements = {
    question,
    columns,
    generateButton,
    status,
    greedySql,
    greedyQuery,
    beamSql,
    beamQuery,
};

const missingElement = Object.entries(elements).find(([, element]) => !element);
if (missingElement) {
    throw new Error(`Missing app element: ${missingElement[0]}`);
}

function renderPrediction(prediction, sqlElement, queryElement) {
    if (!prediction) {
        sqlElement.textContent = 'No prediction was returned.';
        queryElement.textContent = 'No parsed query was returned.';
        return;
    }
    sqlElement.textContent = prediction.sql || 'Could not parse the model output.';
    queryElement.textContent = prediction.query
        ? JSON.stringify(prediction.query, null, 2)
        : 'Could not parse the model output.';
}

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
        if (!response.ok) {
            throw new Error(data.detail || 'The request could not be completed.');
        }

        renderPrediction(data.greedy, greedySql, greedyQuery);
        renderPrediction(data.beam, beamSql, beamQuery);
        status.textContent = 'Complete.';
    } catch (error) {
        status.textContent = error.message;
    } finally {
        generateButton.disabled = false;
    }
});
