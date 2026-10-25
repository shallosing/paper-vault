const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const statusMessage = document.querySelector("#status");
const resultSection = document.querySelector("#result");

form.addEventListener("submit", handleQuestionSubmit);

async function handleQuestionSubmit(event) {
    event.preventDefault();
    resultSection.replaceChildren();
    statusMessage.textContent = "正在检索…";

    try {
        const question = questionInput.value.trim();
        const response = await fetch(`/ask?q=${encodeURIComponent(question)}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "请求失败");
        }

        statusMessage.textContent = `模式：${data.mode}`;
        renderAnswer(data.answer);
        data.citations.forEach(renderCitation);
    } catch (error) {
        statusMessage.textContent = `出错：${error.message}`;
    }
}

function renderAnswer(text) {
    const answer = document.createElement("div");
    answer.className = "answer";
    answer.textContent = text;
    resultSection.append(answer);
}

function renderCitation(source) {
    const card = document.createElement("article");
    card.className = "source";

    const metadata = document.createElement("div");
    metadata.className = "meta";
    metadata.textContent = formatCitationMetadata(source);

    const excerpt = document.createElement("div");
    excerpt.textContent = source.content;

    card.append(metadata, excerpt);
    resultSection.append(card);
}

function formatCitationMetadata(source) {
    const location = source.start_char === null
        ? ""
        : ` · 字符 ${source.start_char}–${source.end_char}`;

    return `[${source.citation_id}] ${source.filename} · 块 ${source.chunk_index}`
        + ` · ${source.source_ref}${location}`;
}
