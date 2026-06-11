let currentMode = 'rag';
let fullGraphData = null;
let cyInstances = [];
let currentSources = [];
let currentTimeCosts = null;
let wikiModalStack = [];

// Conversation management
let conversations = [];
let currentConversationId = null;

// Image upload
let selectedImageFile = null;
let selectedImageObjectUrl = '';

const WIKI_TYPE_COLORS = {
    entity: '#3b82f6',
    concept: '#10b981',
    source: '#f59e0b',
    synthesis: '#8b5cf6',
    query: '#ec4899',
    other: '#6b7280'
};

function makeCyStyle() {
    return [
        {
            selector: 'node',
            style: {
                'shape': 'round-rectangle',
                'width': 'mapData(linkCount, 1, 10, 30, 70)',
                'height': 'mapData(linkCount, 1, 10, 30, 70)',
                'background-color': function(ele) {
                    return WIKI_TYPE_COLORS[ele.data('type')] || WIKI_TYPE_COLORS.other;
                },
                'border-width': 2,
                'border-color': '#ffffff',
                'label': 'data(label)',
                'color': '#1a1a1a',
                'font-size': '12px',
                'font-weight': '600',
                'text-valign': 'center',
                'text-halign': 'center',
                'text-wrap': 'wrap',
                'text-max-width': '80px',
                'text-background-color': 'rgba(255,255,255,0.85)',
                'text-background-opacity': 1,
                'text-background-padding': '2px',
                'text-background-shape': 'roundrectangle'
            }
        },
        {
            selector: 'edge',
            style: {
                'curve-style': 'bezier',
                'width': 'mapData(weight, 0, 10, 1, 4)',
                'line-color': '#9ca3af',
                'target-arrow-color': '#9ca3af',
                'target-arrow-shape': 'triangle',
                'arrow-scale': 0.8,
                'opacity': 0.7
            }
        },
        {
            selector: ':selected',
            style: {
                'background-color': '#ff5722',
                'line-color': '#ff5722',
                'target-arrow-color': '#ff5722'
            }
        },
        {
            selector: '.dimmed',
            style: {
                'opacity': 0.15
            }
        },
        {
            selector: '.highlighted',
            style: {
                'border-color': '#3b82f6',
                'border-width': 3,
                'line-color': '#3b82f6',
                'target-arrow-color': '#3b82f6',
                'opacity': 1
            }
        },
        {
            selector: 'node.highlighted',
            style: {
                'width': 'mapData(linkCount, 1, 10, 38, 82)',
                'height': 'mapData(linkCount, 1, 10, 38, 82)'
            }
        }
    ];
}

// ============================================
// Theme / 主题切换
// ============================================
function initTheme() {
    const savedTheme = localStorage.getItem('rag-wiki-theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('rag-wiki-theme', next);
    updateThemeIcon(next);
}

function updateThemeIcon(theme) {
    const btn = document.getElementById('themeToggle');
    if (btn) btn.textContent = theme === 'light' ? '🌙' : '☀️';
}

// ============================================
// Mode Switching / 模式切换
// ============================================
function initModeSwitching() {
    document.querySelectorAll('.mode-seg-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            setMode(mode);
        });
    });
}

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-seg-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    const wikiConfigBar = document.getElementById('wikiConfigBar');
    if (wikiConfigBar) {
        wikiConfigBar.style.display = mode === 'wiki' ? 'flex' : 'none';
    }
}

// ============================================
// Layout Resizers / 三栏布局拖拽调整
// ============================================
function initLayoutResizers() {
    const layout = document.getElementById('appLayout');
    const leftResizer = document.getElementById('leftLayoutResizer');
    const rightResizer = document.getElementById('rightLayoutResizer');
    if (!layout || !leftResizer || !rightResizer) return;

    let activeResizer = null;

    function startResize(event) {
        activeResizer = event.currentTarget.dataset.resizer;
        event.currentTarget.classList.add('active');
        document.body.classList.add('resizing-layout');
        event.preventDefault();
    }

    function stopResize() {
        activeResizer = null;
        leftResizer.classList.remove('active');
        rightResizer.classList.remove('active');
        document.body.classList.remove('resizing-layout');
    }

    function resize(event) {
        if (!activeResizer || window.innerWidth <= 900) return;
        const currentLeft = parseInt(getComputedStyle(layout).gridTemplateColumns.split(' ')[0], 10) || 260;
        const currentRight = parseInt(getComputedStyle(layout).gridTemplateColumns.split(' ')[4], 10) || 340;
        const minCenter = 420;
        const reserved = minCenter + 12;

        if (activeResizer === 'left') {
            const maxLeft = Math.max(200, window.innerWidth - currentRight - reserved);
            const newWidth = Math.max(180, Math.min(420, event.clientX, maxLeft));
            layout.style.gridTemplateColumns = newWidth + 'px 6px 1fr 6px ' + currentRight + 'px';
        } else {
            const maxRight = Math.max(260, window.innerWidth - currentLeft - reserved);
            const newWidth = Math.max(240, Math.min(520, window.innerWidth - event.clientX, maxRight));
            layout.style.gridTemplateColumns = currentLeft + 'px 6px 1fr 6px ' + newWidth + 'px';
        }
    }

    leftResizer.addEventListener('pointerdown', startResize);
    rightResizer.addEventListener('pointerdown', startResize);
    window.addEventListener('pointermove', resize);
    window.addEventListener('pointerup', stopResize);
    window.addEventListener('pointercancel', stopResize);
}

// ============================================
// Conversation Management / 会话管理
// ============================================
function getWelcomeMarkup() {
    return `
        <div class="message assistant welcome-message">
            <div class="message-content welcome-card">
                <div class="welcome-title">你好！我是 RAG+Wiki 智能助手</div>
                <div class="welcome-desc">可以帮您检索知识库并回答问题。请选择检索模式并开始提问。</div>
            </div>
        </div>
    `;
}

function getEmptyReferenceMarkup() {
    return '<div class="reference-empty">发送问题后，这里会展示 RAG、Wiki 图谱和 Wiki 检索来源。</div>';
}

function formatConversationTime(date) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function saveCurrentConversation() {
    const conversation = conversations.find(item => item.id === currentConversationId);
    if (!conversation) return;
    conversation.messagesHtml = document.getElementById('chatMessages').innerHTML;
    conversation.referenceSummary = document.getElementById('referenceSummary').textContent;
    conversation.referenceHtml = document.getElementById('referenceContent').innerHTML;
    conversation.updatedAt = new Date();
}

function renderConversationList() {
    const list = document.getElementById('conversationList');
    if (!list) return;
    list.innerHTML = '';
    conversations.forEach((conversation) => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'conversation-item' + (conversation.id === currentConversationId ? ' active' : '');
        item.onclick = () => loadConversation(conversation.id);
        item.textContent = conversation.title;

        const time = document.createElement('span');
        time.className = 'conversation-time';
        time.textContent = formatConversationTime(conversation.updatedAt);
        item.appendChild(time);

        list.appendChild(item);
    });
}

function createNewConversation() {
    saveCurrentConversation();

    const now = new Date();
    const conversation = {
        id: 'chat-' + now.getTime(),
        title: '新对话',
        messagesHtml: getWelcomeMarkup(),
        referenceSummary: '发送消息后将在此展示检索来源',
        referenceHtml: getEmptyReferenceMarkup(),
        updatedAt: now
    };

    conversations.unshift(conversation);
    currentConversationId = conversation.id;
    document.getElementById('chatMessages').innerHTML = conversation.messagesHtml;
    document.getElementById('referenceSummary').textContent = conversation.referenceSummary;
    document.getElementById('referenceContent').innerHTML = conversation.referenceHtml;
    clearImage();
    renderConversationList();
    document.getElementById('chatInput').focus();
}

function loadConversation(conversationId) {
    if (conversationId === currentConversationId) return;
    saveCurrentConversation();

    const conversation = conversations.find(item => item.id === conversationId);
    if (!conversation) return;

    currentConversationId = conversation.id;
    document.getElementById('chatMessages').innerHTML = conversation.messagesHtml;
    document.getElementById('referenceSummary').textContent = conversation.referenceSummary;
    document.getElementById('referenceContent').innerHTML = conversation.referenceHtml;
    clearImage();
    renderConversationList();
    document.getElementById('chatInput').focus();
}

function updateCurrentConversationTitle(message) {
    const conversation = conversations.find(item => item.id === currentConversationId);
    if (!conversation) return;
    if (conversation.title === '新对话') {
        conversation.title = message.length > 18 ? message.substring(0, 18) + '...' : message;
    }
    conversation.updatedAt = new Date();
    renderConversationList();
}

// ============================================
// Reference Panel / 右侧面板管理
// ============================================
function setReferenceLoading() {
    document.getElementById('referenceSummary').textContent = '正在检索参考来源...';
    document.getElementById('referenceContent').innerHTML = '<div class="reference-empty">检索中，来源会在结果返回后展示。</div>';
    document.getElementById('referenceGraph').style.display = 'none';
    document.getElementById('referenceBodyResizer').style.display = 'none';
}

function setReferenceError(message) {
    document.getElementById('referenceSummary').textContent = '参考来源获取失败';
    document.getElementById('referenceContent').innerHTML = '<div class="reference-empty">' + escapeHtml(message) + '</div>';
    document.getElementById('referenceGraph').style.display = 'none';
    document.getElementById('referenceBodyResizer').style.display = 'none';
}

function clearReferencePanel() {
    document.getElementById('referenceSummary').textContent = '发送消息后将在此展示检索来源';
    document.getElementById('referenceContent').innerHTML = getEmptyReferenceMarkup();
    document.getElementById('referenceGraph').style.display = 'none';
    document.getElementById('referenceBodyResizer').style.display = 'none';
}

function renderReferencePanel(sources, timeCosts) {
    const contentEl = document.getElementById('referenceContent');
    const summaryEl = document.getElementById('referenceSummary');
    if (!contentEl) return;

    currentSources = sources || [];

    if (!sources || sources.length === 0) {
        contentEl.innerHTML = '<div class="reference-empty">未检索到相关来源</div>';
        if (summaryEl) summaryEl.textContent = '未检索到来源';
        document.getElementById('referenceGraph').style.display = 'none';
        return;
    }

    const ragCount = sources.filter(s => s.origin === 'rag' || !s.origin).length;
    const wikiCount = sources.filter(s => s.origin === 'wiki').length;
    let summaryText = '共 ' + sources.length + ' 条来源';
    if (wikiCount > 0) summaryText += ' (RAG: ' + ragCount + ', Wiki: ' + wikiCount + ')';
    if (summaryEl) summaryEl.textContent = summaryText;

    if (currentMode === 'rag') {
        renderRAGSourceList(contentEl, sources);
    } else {
        renderWikiSourceSplit(contentEl, sources);
    }

    // Embed graph if wiki nodes exist
    const wikiIds = [];
    for (const s of sources) {
        if (s.metadata && s.metadata.source === 'llm_wiki_graph' && s.metadata.wiki_node_id) {
            wikiIds.push(s.metadata.wiki_node_id);
        }
    }
    if (wikiIds.length > 0) {
        renderEmbeddedGraph(wikiIds);
        document.getElementById('referenceBodyResizer').style.display = 'block';
    } else {
        document.getElementById('referenceGraph').style.display = 'none';
        document.getElementById('referenceBodyResizer').style.display = 'none';
    }
}

function renderRAGSourceList(container, sources) {
    container.innerHTML = '';

    const ragSources = sources.filter(s => s.origin === 'rag' || !s.origin);
    if (ragSources.length === 0) {
        container.innerHTML = '<div class="source-panel-empty">未检索到知识库来源</div>';
        return;
    }

    const list = document.createElement('div');
    list.className = 'source-list';

    ragSources.forEach(source => {
        list.appendChild(createSourceListItem(source));
    });

    container.appendChild(list);
}

function renderWikiSourceSplit(container, sources) {
    container.innerHTML = '';

    const wikiSources = sources.filter(s => s.origin === 'wiki');
    const ragSources = sources.filter(s => s.origin === 'rag' || !s.origin);

    const splitDiv = document.createElement('div');
    splitDiv.className = 'source-panel-split';

    // Top: Wiki nodes
    const topDiv = document.createElement('div');
    topDiv.className = 'source-panel-top';

    const topHeader = document.createElement('div');
    topHeader.className = 'source-section-header';
    topHeader.innerHTML = '<span class="source-section-title">Wiki 节点</span><span class="source-section-count">' + wikiSources.length + ' 条</span>';
    topDiv.appendChild(topHeader);

    if (wikiSources.length > 0) {
        const sortedWiki = [...wikiSources].sort((a, b) => {
            const wa = (a.metadata && a.metadata.wiki_relevance) || 0;
            const wb = (b.metadata && b.metadata.wiki_relevance) || 0;
            return wb - wa;
        });
        const wikiList = document.createElement('div');
        wikiList.className = 'source-list';
        sortedWiki.forEach(s => wikiList.appendChild(createSourceListItem(s)));
        topDiv.appendChild(wikiList);
    } else {
        const empty = document.createElement('div');
        empty.className = 'source-panel-empty';
        empty.style.padding = '20px';
        empty.textContent = '无 Wiki 节点';
        topDiv.appendChild(empty);
    }

    // Divider
    const divider = document.createElement('div');
    divider.className = 'source-panel-divider';

    // Bottom: KB sources
    const bottomDiv = document.createElement('div');
    bottomDiv.className = 'source-panel-bottom';

    const bottomHeader = document.createElement('div');
    bottomHeader.className = 'source-section-header';
    bottomHeader.innerHTML = '<span class="source-section-title">知识库来源</span><span class="source-section-count">' + ragSources.length + ' 条</span>';
    bottomDiv.appendChild(bottomHeader);

    if (ragSources.length > 0) {
        const ragList = document.createElement('div');
        ragList.className = 'source-list';
        ragSources.forEach(s => ragList.appendChild(createSourceListItem(s)));
        bottomDiv.appendChild(ragList);
    } else {
        const empty = document.createElement('div');
        empty.className = 'source-panel-empty';
        empty.style.padding = '20px';
        empty.textContent = '无知识库来源';
        bottomDiv.appendChild(empty);
    }

    splitDiv.appendChild(topDiv);
    splitDiv.appendChild(divider);
    splitDiv.appendChild(bottomDiv);
    container.appendChild(splitDiv);

    makeSplitResizable(divider, topDiv, bottomDiv, splitDiv);
}

function makeSplitResizable(divider, topDiv, bottomDiv, container) {
    let isResizing = false;

    divider.addEventListener('mousedown', (e) => {
        isResizing = true;
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const rect = container.getBoundingClientRect();
        const relativeY = e.clientY - rect.top;
        let percent = (relativeY / rect.height) * 100;
        percent = Math.max(15, Math.min(85, percent));
        topDiv.style.flex = '0 0 ' + percent + '%';
        bottomDiv.style.flex = '1';
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

function initReferenceBodyResizer() {
    const resizer = document.getElementById('referenceBodyResizer');
    const content = document.getElementById('referenceContent');
    const graph = document.getElementById('referenceGraph');
    const body = document.querySelector('.reference-body');
    if (!resizer || !content || !graph || !body) return;

    let isResizing = false;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        resizer.classList.add('active');
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const rect = body.getBoundingClientRect();
        const relativeY = e.clientY - rect.top;
        let percent = (relativeY / rect.height) * 100;
        percent = Math.max(20, Math.min(80, percent));
        content.style.flex = '0 0 ' + percent + '%';
        graph.style.flex = '1';
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizer.classList.remove('active');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

function createSourceListItem(source) {
    const item = document.createElement('div');
    item.className = 'source-list-item';

    const isWiki = source.metadata && source.metadata.source === 'llm_wiki_graph';

    // Score badge
    const scoreBadge = document.createElement('div');
    scoreBadge.className = 'source-list-score ' + (isWiki ? 'wiki' : 'rag');

    if (isWiki) {
        const relevance = (source.metadata && source.metadata.wiki_relevance) || 0;
        scoreBadge.textContent = relevance.toFixed(1);
    } else {
        const score = scaleRagScore(source.score || 0);
        scoreBadge.textContent = score + '%';
    }

    // Text preview
    const textDiv = document.createElement('div');
    textDiv.className = 'source-list-text';
    if (source.metadata && source.metadata.wiki_node_title) {
        textDiv.textContent = source.metadata.wiki_node_title;
    } else if (source.metadata && source.metadata.file_name) {
        textDiv.textContent = source.metadata.file_name;
    } else {
        textDiv.textContent = (source.content || '').substring(0, 60) + '...';
    }

    item.appendChild(scoreBadge);
    item.appendChild(textDiv);

    item.addEventListener('click', (e) => {
        e.stopPropagation();
        showSourceDetail(source);
    });
    return item;
}

function scaleRagScore(originalScore) {
    const clamped = Math.max(0, Math.min(1, originalScore));
    return Math.round(50 + clamped * 50);
}

// ============================================
// Markdown Renderer / 轻量级 Markdown 渲染
// ============================================
function renderMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    // Code blocks
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, function(match, lang, code) {
        return '<pre><code>' + code.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</code></pre>';
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers
    html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold & Italic
    html = html.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // Blockquotes
    html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

    // Unordered lists
    html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
    html = html.replace(/^(<li>.*<\/li>\n?)+/gim, function(match) {
        return '<ul>' + match.trim() + '</ul>';
    });

    // Ordered lists
    html = html.replace(/^\d+\.\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/^(<li>.*<\/li>\n?)+/gim, function(match) {
        if (!match.includes('<ul>')) return '<ol>' + match.trim() + '</ol>';
        return match;
    });

    // Paragraphs (simple: split by double newline)
    const blocks = html.split(/\n\n+/);
    html = blocks.map(block => {
        block = block.trim();
        if (!block) return '';
        if (block.startsWith('<')) return block;
        return '<p>' + block.replace(/\n/g, '<br>') + '</p>';
    }).join('');

    return html;
}

// ============================================
// Image Upload / 图片上传
// ============================================
function handleImageChange(event) {
    const file = event.target.files && event.target.files[0] ? event.target.files[0] : null;
    if (!file) {
        clearImage();
        return;
    }
    if (!file.type.startsWith('image/')) {
        clearImage();
        alert('请选择图片文件。');
        return;
    }
    if (selectedImageObjectUrl) {
        URL.revokeObjectURL(selectedImageObjectUrl);
    }
    selectedImageFile = file;
    selectedImageObjectUrl = URL.createObjectURL(file);

    const preview = document.getElementById('imagePreview');
    const wrap = document.getElementById('imagePreviewWrap');
    preview.src = selectedImageObjectUrl;
    preview.alt = file.name || '图片预览';
    wrap.style.display = 'inline-flex';
}

function clearImage() {
    if (selectedImageObjectUrl) {
        URL.revokeObjectURL(selectedImageObjectUrl);
        selectedImageObjectUrl = '';
    }
    selectedImageFile = null;
    document.getElementById('imageInput').value = '';
    const preview = document.getElementById('imagePreview');
    const wrap = document.getElementById('imagePreviewWrap');
    preview.removeAttribute('src');
    wrap.style.display = 'none';
}

function readImageAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '');
        reader.onerror = () => reject(reader.error || new Error('读取图片失败'));
        reader.readAsDataURL(file);
    });
}

function extractBase64(dataUrl) {
    return typeof dataUrl === 'string' && dataUrl.includes(',')
        ? dataUrl.split(',')[1]
        : '';
}

// ============================================
// Chat / 聊天功能
// ============================================
function sendMessage() {
    const input = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    const message = input.value.trim();

    if (!message) return;

    closeWikiPanel();
    sendButton.disabled = true;
    sendButton.innerHTML = '<div class="loading"></div>';

    let imageDataUrl = '';
    let imageBase64 = '';
    let imageName = '';

    const processAndSend = async () => {
        if (selectedImageFile) {
            try {
                imageName = selectedImageFile.name;
                imageDataUrl = await readImageAsDataUrl(selectedImageFile);
                imageBase64 = extractBase64(imageDataUrl);
            } catch (error) {
                console.error('读取图片失败:', error);
                addMessage('assistant', '抱歉，图片读取失败：' + error.message);
                resetSendButton();
                return;
            }
        }

        // Add user message
        addMessage('user', message, imageDataUrl ? { url: imageDataUrl, name: imageName } : null);
        updateCurrentConversationTitle(message);
        setReferenceLoading();
        if (imageDataUrl) {
            clearImage();
        }
        input.value = '';

        const typingId = addTypingIndicator();
        const relevanceValue = parseFloat(document.getElementById('relevanceSlider').value);

        try {
            const response = await fetch('/query_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: message,
                    parseType: 'firmware',
                    use_llm: true,
                    retrieval_mode: currentMode === 'wiki' ? 'all' : 'rag',
                    wiki_min_relevance: relevanceValue,
                    image_base64: imageBase64,
                    image_name: imageName
                })
            });

            if (!response.ok) {
                throw new Error('网络响应异常: ' + response.status);
            }

            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content markdown-content';
            contentDiv.id = 'streaming-' + typingId;
            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);

            removeTypingIndicator(typingId);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let sources = [];
            let fullAnswer = '';
            let timeCosts = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));

                            if (data.type === 'sources') {
                                sources = data.data || [];
                                timeCosts = data.time_cost || null;
                                renderReferencePanel(sources, timeCosts);
                                saveCurrentConversation();
                            } else if (data.type === 'content') {
                                fullAnswer += data.content;
                                const displayText = getDisplayableText(fullAnswer);
                                contentDiv.innerHTML = renderMarkdown(displayText);
                                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                            } else if (data.type === 'done') {
                                if (data.time_cost) {
                                    timeCosts = { ...data.time_cost, total_time: data.total_time };
                                }
                            } else if (data.type === 'error') {
                                fullAnswer += '\n\n[错误: ' + data.error + ']';
                                contentDiv.innerHTML = renderMarkdown(fullAnswer);
                                resetSendButton();
                                return;
                            }
                        } catch (e) {
                            console.error('解析SSE数据失败:', e, line);
                        }
                    }
                }
            }

            finalizeAnswer(messageDiv, contentDiv, sources, timeCosts, fullAnswer);
        } catch (error) {
            removeTypingIndicator(typingId);
            addMessage('assistant', '抱歉，发生了错误：' + error.message);
            resetSendButton();
        }
    };

    processAndSend();
}

function finalizeAnswer(messageDiv, contentDiv, sources, timeCosts, fullAnswer) {
    const cleanAnswer = stripAnswerPrefix(fullAnswer);
    contentDiv.innerHTML = renderMarkdown(cleanAnswer);

    if (sources && sources.length > 0) {
        renderReferencePanel(sources, timeCosts);
    }
    if (timeCosts) {
        addTimeInfoButton(messageDiv, timeCosts);
    }
    saveCurrentConversation();
    resetSendButton();
}

function addWikiGraphButton(messageDiv, sources) {
    const wikiIds = [];
    for (const s of sources) {
        if (s.metadata && s.metadata.source === 'llm_wiki_graph' && s.metadata.wiki_node_id) {
            wikiIds.push(s.metadata.wiki_node_id);
        }
    }
    if (wikiIds.length === 0) return;

    const btn = document.createElement('button');
    btn.className = 'wiki-graph-btn';
    btn.textContent = '\u5c55\u793a Wiki \u77e5\u8bc6\u56fe\u8c31';
    btn.addEventListener('click', () => {
        showWikiGraph(wikiIds);
    });
    messageDiv.appendChild(btn);
}

function resetSendButton() {
    const sendButton = document.getElementById('sendButton');
    const input = document.getElementById('chatInput');
    sendButton.disabled = false;
    sendButton.innerHTML = '发送';
    input.focus();
}

function stripAnswerPrefix(text) {
    return text.replace(/^(答案[：:]?\s*)/i, '').trim();
}

const PREFIX_VARIANTS = ['答案：', '答案:', '答案'];

function getDisplayableText(text) {
    for (const v of PREFIX_VARIANTS) {
        if (v.startsWith(text)) return '';
    }
    for (const v of PREFIX_VARIANTS) {
        if (text.startsWith(v)) return text.slice(v.length);
    }
    return text;
}

function addMessage(role, content, imageInfo = null) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ' + role;

    if (imageInfo && imageInfo.url) {
        const imgWrap = document.createElement('div');
        imgWrap.className = 'message-image-wrap';
        const img = document.createElement('img');
        img.className = 'message-image';
        img.src = imageInfo.url;
        img.alt = imageInfo.name || '图片';
        imgWrap.appendChild(img);
        messageDiv.appendChild(imgWrap);
    }

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    messageDiv.appendChild(contentDiv);

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addTypingIndicator() {
    const messagesDiv = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    const typingId = 'typing-' + Date.now();
    typingDiv.id = typingId;
    typingDiv.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return typingId;
}

function removeTypingIndicator(typingId) {
    const el = document.getElementById(typingId);
    if (el) el.remove();
}

// ============================================
// Source Detail Popup / 来源详情弹窗
// ============================================
function showSourceDetail(source) {
    const isWiki = source.metadata && source.metadata.source === 'llm_wiki_graph';
    if (isWiki) {
        openWikiNodeModal(source);
    } else {
        showRAGSourcePopup(source);
    }
}

function showRAGSourcePopup(source) {
    try {
        const popup = document.getElementById('sourcePopup');
        const titleEl = document.getElementById('sourcePopupTitle');
        const bodyEl = document.getElementById('sourcePopupBody');
        if (!popup || !titleEl || !bodyEl) return;

        const title = source.metadata && source.metadata.file_name
            ? source.metadata.file_name
            : (source.source || '来源详情');
        titleEl.textContent = title;

        let html = '';

        // Score
        if (source.score !== undefined) {
            html += '<div class="detail-row">';
            html += '<div class="detail-label">相关度得分</div>';
            html += '<div class="detail-content">' + scaleRagScore(source.score) + '%</div>';
            html += '</div>';
        }

        // Content
        if (source.content) {
            html += '<div class="detail-row">';
            html += '<div class="detail-label">内容</div>';
            let contentHtml = '';
            if (typeof marked !== 'undefined') {
                contentHtml = marked.parse(source.content);
            } else {
                contentHtml = escapeHtml(source.content);
            }
            html += '<div class="detail-content markdown-body">' + contentHtml + '</div>';
            html += '</div>';
        }

        // Metadata
        if (source.metadata && Object.keys(source.metadata).length > 0) {
            html += '<div class="detail-row">';
            html += '<div class="detail-label">元数据</div>';
            html += '<table class="meta-table">';
            for (const [key, value] of Object.entries(source.metadata)) {
                if (key === 'embedding' || key === 'vector') continue;
                let displayValue = value;
                if (typeof value === 'object') {
                    displayValue = JSON.stringify(value, null, 2);
                }
                if (typeof displayValue === 'string' && displayValue.length > 500) {
                    displayValue = displayValue.substring(0, 500) + '...';
                }
                html += '<tr><td>' + escapeHtml(key) + '</td><td>' + escapeHtml(String(displayValue)) + '</td></tr>';
            }
            html += '</table>';
            html += '</div>';
        }

        bodyEl.innerHTML = html;
        popup.style.display = 'flex';
    } catch (e) {
        console.error('展示详情失败:', e);
    }
}

function closeSourcePopup() {
    const popup = document.getElementById('sourcePopup');
    if (popup) popup.style.display = 'none';
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// Wiki Node Modal / Wiki 节点详情弹窗
// ============================================
function parseFrontmatter(content) {
    const match = content.match(/^---\s*\n([\s\S]*?)\n---\s*\n?/);
    if (!match) return { frontmatter: {}, body: content };

    const yamlText = match[1];
    const body = content.substring(match[0].length);
    const frontmatter = {};

    let currentKey = null;
    yamlText.split('\n').forEach(line => {
        const idx = line.indexOf(':');
        if (idx > 0 && !line.trim().startsWith('-')) {
            currentKey = line.substring(0, idx).trim();
            let value = line.substring(idx + 1).trim();
            if ((value.startsWith('"') && value.endsWith('"')) ||
                (value.startsWith("'") && value.endsWith("'"))) {
                value = value.slice(1, -1);
            }
            frontmatter[currentKey] = value;
        } else if (line.trim().startsWith('-') && currentKey) {
            let item = line.trim().substring(1).trim();
            if ((item.startsWith('"') && item.endsWith('"')) ||
                (item.startsWith("'") && item.endsWith("'"))) {
                item = item.slice(1, -1);
            }
            if (!Array.isArray(frontmatter[currentKey])) {
                frontmatter[currentKey] = [frontmatter[currentKey]];
            }
            frontmatter[currentKey].push(item);
        }
    });

    return { frontmatter, body };
}

function openWikiNodeModal(source) {
    const content = source.content || '';
    const metadata = source.metadata || {};
    const nodeId = metadata.wiki_node_id || metadata.file_name || '';
    const title = metadata.wiki_node_title || nodeId;

    const parsed = parseFrontmatter(content);

    wikiModalStack = [{
        nodeId: nodeId,
        title: title,
        type: metadata.wiki_node_type || 'unknown',
        content: content,
        frontmatter: parsed.frontmatter,
        body: parsed.body,
        relatedNodes: null
    }];

    renderWikiNodeModal();
    fetchRelatedForCurrentNode();
}

async function openWikiNodeModalById(nodeId, title) {
    try {
        const resp = await fetch('/wiki_node_related', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: nodeId,
                hops: 2,
                min_relevance: 2.0,
                decay: 0.7,
                limit: 10
            })
        });
        const result = await resp.json();
        if (result.code !== 0) {
            console.error('获取相关节点失败:', result.msg);
            return;
        }
        const data = result.data;
        wikiModalStack.push({
            nodeId: data.node_id,
            title: data.node_title,
            type: data.node_type,
            content: data.node_content,
            frontmatter: data.node_frontmatter || {},
            body: data.node_content,
            relatedNodes: data.related_nodes
        });
        renderWikiNodeModal();
    } catch (e) {
        console.error('打开 Wiki 节点弹窗失败:', e);
    }
}

function renderWikiNodeModal() {
    const modal = document.getElementById('wikiNodeModal');
    if (!modal || wikiModalStack.length === 0) return;

    const current = wikiModalStack[wikiModalStack.length - 1];

    document.getElementById('wikiNodeTitle').textContent = current.title;
    document.getElementById('wikiNodeType').textContent = current.type || 'unknown';

    // Breadcrumb
    const bcEl = document.getElementById('wikiNodeBreadcrumb');
    let bcHtml = '';
    wikiModalStack.forEach((entry, idx) => {
        if (idx > 0) bcHtml += '<span class="bc-sep">></span>';
        if (idx === wikiModalStack.length - 1) {
            bcHtml += '<span>' + escapeHtml(entry.title) + '</span>';
        } else {
            bcHtml += '<a onclick="goToBreadcrumb(' + idx + ', event)">' + escapeHtml(entry.title) + '</a>';
        }
    });
    bcEl.innerHTML = bcHtml;

    // Frontmatter bar — 隐藏 tags/sources/created，只展示其余有效 frontmatter
    const fmBar = document.getElementById('wikiFrontmatterBar');
    let fmHtml = '';
    const fm = current.frontmatter || {};
    const hiddenKeys = new Set(['tags', 'sources', 'created']);
    const displayKeys = ['author', 'version', 'status', 'date'];
    for (const key of displayKeys) {
        if (hiddenKeys.has(key)) continue;
        if (fm[key] !== undefined && fm[key] !== null && fm[key] !== '') {
            fmHtml += '<div class="fm-item"><span class="fm-key">' + escapeHtml(key) + ':</span> <span class="fm-value">' + escapeHtml(String(fm[key])) + '</span></div>';
        }
    }
    if (fmHtml) {
        fmBar.style.display = 'block';
        fmBar.innerHTML = fmHtml;
    } else {
        fmBar.style.display = 'none';
    }

    // Content
    const contentEl = document.getElementById('wikiNodeContent');
    const bodyText = current.body || current.content || '';
    if (typeof marked !== 'undefined') {
        contentEl.innerHTML = marked.parse(bodyText);
    } else {
        contentEl.textContent = bodyText;
    }

    // Related nodes
    const relatedSection = document.getElementById('wikiRelatedSection');
    const relatedCards = document.getElementById('wikiRelatedCards');

    if (current.relatedNodes && current.relatedNodes.length > 0) {
        relatedSection.style.display = 'block';
        relatedCards.innerHTML = '';
        current.relatedNodes.forEach(node => {
            const card = document.createElement('div');
            card.className = 'wiki-related-card';
            card.innerHTML = '<div class="rc-title">' + escapeHtml(node.title) + '</div>' +
                '<div class="rc-meta"><span>' + escapeHtml(node.type || 'unknown') + '</span>' +
                '<span class="rc-score">' + (node.effective_relevance || 0).toFixed(1) + '</span></div>';
            card.addEventListener('click', () => {
                openWikiNodeModalById(node.id, node.title);
            });
            relatedCards.appendChild(card);
        });
    } else if (current.relatedNodes === null) {
        relatedSection.style.display = 'block';
        relatedCards.innerHTML = '<div style="color:var(--text-tertiary);font-size:12px;padding:8px 0;">加载相关节点中...</div>';
    } else {
        relatedSection.style.display = 'none';
    }

    modal.style.display = 'flex';
}

async function fetchRelatedForCurrentNode() {
    if (wikiModalStack.length === 0) return;
    const current = wikiModalStack[wikiModalStack.length - 1];
    if (current.relatedNodes !== null) return;

    try {
        const resp = await fetch('/wiki_node_related', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: current.nodeId,
                hops: 2,
                min_relevance: 2.0,
                decay: 0.7,
                limit: 10
            })
        });
        const result = await resp.json();
        if (result.code === 0) {
            current.relatedNodes = result.data.related_nodes;
            renderWikiNodeModal();
        } else {
            current.relatedNodes = [];
            renderWikiNodeModal();
        }
    } catch (e) {
        console.error('获取相关节点失败:', e);
        current.relatedNodes = [];
    }
}

function goToBreadcrumb(index, event) {
    if (event) event.stopPropagation();
    if (index < 0 || index >= wikiModalStack.length - 1) return;
    wikiModalStack = wikiModalStack.slice(0, index + 1);
    renderWikiNodeModal();
}

function closeWikiNodeModal() {
    const modal = document.getElementById('wikiNodeModal');
    if (modal) modal.style.display = 'none';
    wikiModalStack = [];
}

// ============================================
// Time Info / 耗时信息
// ============================================
function addTimeInfoButton(messageDiv, timeCosts) {
    const btn = document.createElement('button');
    btn.className = 'time-info-btn';
    const total = timeCosts.total_time || timeCosts.total || 0;
    btn.textContent = '\u23f1 耗时详情 (' + total.toFixed(2) + 's)';
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        showTimePopup(timeCosts);
    });
    messageDiv.appendChild(btn);
}

function showTimePopup(timeCosts) {
    let popup = document.getElementById('timePopup');
    if (!popup) {
        popup = document.createElement('div');
        popup.className = 'time-popup';
        popup.id = 'timePopup';
        document.body.appendChild(popup);
    }

    const labelMap = {
        'rag_retrieval': 'RAG 检索',
        'table_recovery': '表格复原',
        'concat_context': '拼接上下文',
        'wiki_search': 'Wiki 独立检索',
        'rrf_fusion': 'RRF 融合',
        'external_rerank': '外部 Rerank',
        'wiki_enhance': 'Wiki 增强与 RRF',
        'llm_generation': '答案生成',
        'retrieval': '检索总耗时',
        'merge': '结果合并',
        'total': '总耗时',
        'total_time': '总耗时'
    };

    let html = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">';
    html += '<strong>\u23f1 耗时明细</strong>';
    html += '<span style="cursor:pointer;color:var(--text-tertiary);font-size:18px;">&times;</span>';
    html += '</div>';
    html += '<table>';

    for (const [key, value] of Object.entries(timeCosts)) {
        if (key === 'total_time' && timeCosts['total']) continue;
        const label = labelMap[key] || key;
        const val = typeof value === 'number' ? value.toFixed(3) + ' s' : String(value);
        html += '<tr><td>' + label + '</td><td>' + val + '</td></tr>';
    }

    html += '</table>';
    popup.innerHTML = html;
    popup.style.display = 'block';

    popup.querySelector('span').addEventListener('click', () => {
        popup.style.display = 'none';
    });

    setTimeout(() => {
        const closeHandler = (ev) => {
            if (!popup.contains(ev.target)) {
                popup.style.display = 'none';
                document.removeEventListener('click', closeHandler);
            }
        };
        document.addEventListener('click', closeHandler);
    }, 0);
}

// ============================================
// Wiki Graph / Wiki 知识图谱
// ============================================
async function showWikiGraph(wikiNodeIds) {
    if (!wikiNodeIds || wikiNodeIds.length === 0) return;

    const panel = document.getElementById('wikiPanel');
    const clustersContainer = document.getElementById('wikiClusters');
    clustersContainer.innerHTML = '';

    cyInstances.forEach(cy => cy.destroy());
    cyInstances = [];

    try {
        if (!fullGraphData) {
            const resp = await fetch('/wiki_graph');
            const result = await resp.json();
            if (result.code === 0) {
                fullGraphData = result.data;
            } else {
                console.error('获取Wiki图谱失败:', result.msg);
                return;
            }
        }

        const nodeIdSet = new Set(wikiNodeIds);
        const nodeMap = {};
        fullGraphData.nodes.forEach(n => { nodeMap[n.id] = n; });

        fullGraphData.edges.forEach(e => {
            if (nodeIdSet.has(e.source) || nodeIdSet.has(e.target)) {
                nodeIdSet.add(e.source);
                nodeIdSet.add(e.target);
            }
        });

        const subNodes = fullGraphData.nodes.filter(n => nodeIdSet.has(n.id));
        const subEdges = fullGraphData.edges.filter(e => nodeIdSet.has(e.source) && nodeIdSet.has(e.target));

        if (subNodes.length === 0) return;

        const clusters = groupIntoClusters(subNodes, subEdges);

        clusters.forEach((cluster, idx) => {
            const box = document.createElement('div');
            box.className = 'cluster-box';

            const header = document.createElement('div');
            header.className = 'cluster-header';
            const repTitle = cluster.nodes[0] ? cluster.nodes[0].title : '\u7c07 ' + (idx + 1);
            header.innerHTML = '<span>' + repTitle + ' (' + cluster.nodes.length + ' \u8282\u70b9)</span>' +
                '<div class="cluster-actions">' +
                '<button onclick="resetClusterView(' + idx + ')">\u91cd\u7f6e\u89c6\u89d2</button>' +
                '</div>';
            box.appendChild(header);

            const canvasId = 'cluster-canvas-' + idx;
            const canvasDiv = document.createElement('div');
            canvasDiv.className = 'cluster-canvas';
            canvasDiv.id = canvasId;
            box.appendChild(canvasDiv);

            box.addEventListener('click', (e) => {
                if (e.target.closest('.cluster-actions') || e.target.closest('.cluster-canvas')) return;
                expandClusterToOverlay(cluster, idx, elements, clusterColor);
            });

            clustersContainer.appendChild(box);

            const elements = [];
            cluster.nodes.forEach(n => {
                elements.push({
                    data: {
                        id: n.id,
                        label: n.title || n.id,
                        linkCount: n.linkCount || 1,
                        type: n.type || 'unknown',
                        path: n.path || '',
                        sources: n.sources || []
                    }
                });
            });
            cluster.edges.forEach(e => {
                elements.push({
                    data: {
                        source: e.source,
                        target: e.target,
                        weight: e.weight || 1
                    }
                });
            });

            const colors = ['#667eea', '#764ba2', '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];
            const clusterColor = colors[idx % colors.length];

            const cy = cytoscape({
                container: document.getElementById(canvasId),
                elements: elements,
                style: makeCyStyle(),
                layout: {
                    name: 'cose',
                    padding: 10,
                    nodeRepulsion: 400000,
                    edgeElasticity: 100,
                    nestingFactor: 5,
                    gravity: 80,
                    numIter: 1000,
                    initialTemp: 200,
                    coolingFactor: 0.95,
                    minTemp: 1.0
                },
                wheelSensitivity: 0.3,
                userZoomingEnabled: true,
                userPanningEnabled: true,
                boxSelectionEnabled: false,
                minZoom: 0.3,
                maxZoom: 3
            });

            cy.on('tap', 'node', function(evt) {
                const node = evt.target;
                const pos = evt.originalEvent ? { x: evt.originalEvent.clientX, y: evt.originalEvent.clientY } : null;
                showNodePopup(node, pos);
            });

            cy.on('mouseover', 'node', function(evt) {
                const node = evt.target;
                const neighborhood = node.neighborhood().add(node);
                cy.elements().not(neighborhood).addClass('dimmed');
                neighborhood.addClass('highlighted');
            });

            cy.on('mouseout', 'node', function(evt) {
                cy.elements().removeClass('dimmed highlighted');
            });

            cyInstances.push(cy);
        });

        panel.style.display = 'flex';
    } catch (e) {
        console.error('展示Wiki图谱失败:', e);
    }
}

function groupIntoClusters(nodes, edges) {
    const hasCommunity = nodes.some(n => n.community !== undefined);
    if (hasCommunity) {
        const groups = {};
        nodes.forEach(n => {
            const c = n.community !== undefined ? n.community : 0;
            if (!groups[c]) groups[c] = { nodes: [], edges: [] };
            groups[c].nodes.push(n);
        });
        const nodeIdsInGroup = {};
        Object.keys(groups).forEach(key => {
            nodeIdsInGroup[key] = new Set(groups[key].nodes.map(n => n.id));
        });
        edges.forEach(e => {
            Object.keys(groups).forEach(key => {
                if (nodeIdsInGroup[key].has(e.source) && nodeIdsInGroup[key].has(e.target)) {
                    groups[key].edges.push(e);
                }
            });
        });
        return Object.values(groups).filter(g => g.nodes.length >= 2);
    }

    const adj = {};
    nodes.forEach(n => { adj[n.id] = []; });
    edges.forEach(e => {
        if (adj[e.source]) adj[e.source].push(e.target);
        if (adj[e.target]) adj[e.target].push(e.source);
    });

    const visited = new Set();
    const components = [];

    function dfs(nodeId, component) {
        visited.add(nodeId);
        component.push(nodeId);
        for (const neighbor of (adj[nodeId] || [])) {
            if (!visited.has(neighbor)) {
                dfs(neighbor, component);
            }
        }
    }

    nodes.forEach(n => {
        if (!visited.has(n.id)) {
            const component = [];
            dfs(n.id, component);
            components.push(component);
        }
    });

    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });

    const clusters = components.map(comp => {
        const compSet = new Set(comp);
        return {
            nodes: comp.map(id => nodeMap[id]).filter(Boolean),
            edges: edges.filter(e => compSet.has(e.source) && compSet.has(e.target))
        };
    });
    return clusters.filter(c => c.nodes.length >= 2);
}

function resetClusterView(idx) {
    if (cyInstances[idx]) {
        cyInstances[idx].fit();
        cyInstances[idx].center();
    }
}

function closeWikiPanel() {
    const panel = document.getElementById('wikiPanel');
    if (panel) panel.style.display = 'none';
    cyInstances.forEach(cy => cy.destroy());
    cyInstances = [];
}

// ============================================
// Embedded Graph / 右侧面板嵌入知识图谱
// ============================================
let embeddedCyInstances = [];
let embeddedClusterData = [];

async function renderEmbeddedGraph(wikiNodeIds) {
    if (!wikiNodeIds || wikiNodeIds.length === 0) return;

    const graphContainer = document.getElementById('referenceGraph');
    const navEl = document.getElementById('graphNav');
    const viewportEl = document.getElementById('graphViewport');

    // Clean up previous embedded instances
    embeddedCyInstances.forEach(cy => cy.destroy());
    embeddedCyInstances = [];
    embeddedClusterData = [];
    viewportEl.innerHTML = '';
    navEl.innerHTML = '';

    try {
        if (!fullGraphData) {
            const resp = await fetch('/wiki_graph');
            const result = await resp.json();
            if (result.code === 0) {
                fullGraphData = result.data;
            } else {
                console.error('获取Wiki图谱失败:', result.msg);
                return;
            }
        }

        const nodeIdSet = new Set(wikiNodeIds);
        fullGraphData.edges.forEach(e => {
            if (nodeIdSet.has(e.source) || nodeIdSet.has(e.target)) {
                nodeIdSet.add(e.source);
                nodeIdSet.add(e.target);
            }
        });

        const subNodes = fullGraphData.nodes.filter(n => nodeIdSet.has(n.id));
        const subEdges = fullGraphData.edges.filter(e => nodeIdSet.has(e.source) && nodeIdSet.has(e.target));

        if (subNodes.length === 0) {
            graphContainer.style.display = 'none';
            return;
        }

        const clusters = groupIntoClusters(subNodes, subEdges);
        if (clusters.length === 0) {
            graphContainer.style.display = 'none';
            return;
        }

        graphContainer.style.display = 'flex';
        embeddedClusterData = clusters;

        // Build nav
        clusters.forEach((cluster, idx) => {
            const btn = document.createElement('button');
            const repTitle = cluster.nodes[0] ? cluster.nodes[0].title : '簇 ' + (idx + 1);
            btn.textContent = repTitle + ' (' + cluster.nodes.length + ')';
            btn.dataset.index = idx;
            if (idx === 0) btn.classList.add('active');
            btn.addEventListener('click', () => switchEmbeddedCluster(idx));
            navEl.appendChild(btn);
        });

        // Expand button
        const expandBtn = document.createElement('button');
        expandBtn.className = 'graph-expand-btn';
        expandBtn.textContent = '\u26f6 放大';
        expandBtn.title = '在独立窗口中查看知识图谱';
        expandBtn.addEventListener('click', () => showWikiGraph(wikiNodeIds));
        navEl.appendChild(expandBtn);

        clusters.forEach((cluster, idx) => {
            // Create canvas container
            const canvasId = 'embedded-cluster-' + idx;
            const canvasDiv = document.createElement('div');
            canvasDiv.className = 'embedded-cluster-canvas';
            canvasDiv.id = canvasId;
            canvasDiv.style.display = idx === 0 ? 'block' : 'none';
            canvasDiv.style.width = '100%';
            canvasDiv.style.height = '100%';
            viewportEl.appendChild(canvasDiv);

            const elements = [];
            cluster.nodes.forEach(n => {
                elements.push({
                    data: {
                        id: n.id,
                        label: n.title || n.id,
                        linkCount: n.linkCount || 1,
                        type: n.type || 'unknown',
                        path: n.path || '',
                        sources: n.sources || []
                    }
                });
            });
            cluster.edges.forEach(e => {
                elements.push({
                    data: {
                        source: e.source,
                        target: e.target,
                        weight: e.weight || 1
                    }
                });
            });

            const cy = cytoscape({
                container: document.getElementById(canvasId),
                elements: elements,
                style: makeCyStyle(),
                layout: {
                    name: 'cose',
                    padding: 8,
                    nodeRepulsion: 400000,
                    edgeElasticity: 100,
                    nestingFactor: 5,
                    gravity: 80,
                    numIter: 1000,
                    initialTemp: 200,
                    coolingFactor: 0.95,
                    minTemp: 1.0
                },
                wheelSensitivity: 0.3,
                userZoomingEnabled: true,
                userPanningEnabled: true,
                boxSelectionEnabled: false,
                minZoom: 0.3,
                maxZoom: 3
            });

            cy.on('tap', 'node', function(evt) {
                const node = evt.target;
                const pos = evt.originalEvent ? { x: evt.originalEvent.clientX, y: evt.originalEvent.clientY } : null;
                showNodePopup(node, pos);
            });

            cy.on('mouseover', 'node', function(evt) {
                const node = evt.target;
                const neighborhood = node.neighborhood().add(node);
                cy.elements().not(neighborhood).addClass('dimmed');
                neighborhood.addClass('highlighted');
            });

            cy.on('mouseout', 'node', function(evt) {
                cy.elements().removeClass('dimmed highlighted');
            });

            embeddedCyInstances.push(cy);
        });
    } catch (e) {
        console.error('嵌入Wiki图谱失败:', e);
        graphContainer.style.display = 'none';
    }
}

function switchEmbeddedCluster(index) {
    const navEl = document.getElementById('graphNav');
    const buttons = navEl.querySelectorAll('button');
    buttons.forEach((btn, idx) => {
        btn.classList.toggle('active', idx === index);
    });

    embeddedCyInstances.forEach((cy, idx) => {
        const container = cy.container();
        if (container) container.style.display = idx === index ? 'block' : 'none';
    });
}

function expandClusterToOverlay(cluster, idx, elements, clusterColor) {
    const overlay = document.createElement('div');
    overlay.className = 'cluster-overlay';
    overlay.id = 'clusterOverlay-' + idx;

    const box = document.createElement('div');
    box.className = 'cluster-overlay-box';

    const header = document.createElement('div');
    header.className = 'cluster-overlay-header';
    const repTitle = cluster.nodes[0] ? cluster.nodes[0].title : '\u7c07 ' + (idx + 1);
    header.innerHTML = '<span>' + repTitle + ' (' + cluster.nodes.length + ' \u8282\u70b9)</span>' +
        '<button onclick="closeClusterOverlay(' + idx + ')">\u00d7</button>';
    box.appendChild(header);

    const canvasId = 'cluster-overlay-canvas-' + idx;
    const canvasDiv = document.createElement('div');
    canvasDiv.className = 'cluster-overlay-canvas';
    canvasDiv.id = canvasId;
    box.appendChild(canvasDiv);

    overlay.appendChild(box);
    document.body.appendChild(overlay);

    const cy = cytoscape({
        container: document.getElementById(canvasId),
        elements: elements,
        style: makeCyStyle(),
        layout: {
            name: 'cose',
            padding: 10,
            nodeRepulsion: 400000,
            edgeElasticity: 100,
            nestingFactor: 5,
            gravity: 80,
            numIter: 1000,
            initialTemp: 200,
            coolingFactor: 0.95,
            minTemp: 1.0
        },
        wheelSensitivity: 0.3,
        userZoomingEnabled: true,
        userPanningEnabled: true,
        boxSelectionEnabled: false,
        minZoom: 0.3,
        maxZoom: 3
    });

    cy.on('tap', 'node', function(evt) {
        const node = evt.target;
        const pos = evt.originalEvent ? { x: evt.originalEvent.clientX, y: evt.originalEvent.clientY } : null;
        showNodePopup(node, pos);
    });

    cy.on('mouseover', 'node', function(evt) {
        const node = evt.target;
        const neighborhood = node.neighborhood().add(node);
        cy.elements().not(neighborhood).addClass('dimmed');
        neighborhood.addClass('highlighted');
    });

    cy.on('mouseout', 'node', function(evt) {
        cy.elements().removeClass('dimmed highlighted');
    });

    makeDraggable(box, header);
}

function closeClusterOverlay(idx) {
    const overlay = document.getElementById('clusterOverlay-' + idx);
    if (overlay) overlay.remove();
}

function showNodePopup(node, mousePos) {
    const popup = document.getElementById('nodePopup');
    const titleEl = document.getElementById('popupTitle');
    const contentEl = document.getElementById('popupContent');
    if (!popup || !titleEl || !contentEl) return;

    titleEl.textContent = node.data('label') || node.data('id');

    let html = '';
    html += '<div><strong>\u7c7b\u578b:</strong> ' + (node.data('type') || 'unknown') + '</div>';
    html += '<div><strong>\u8fde\u63a5\u6570:</strong> ' + (node.data('linkCount') || 0) + '</div>';
    if (node.data('path')) {
        html += '<div><strong>\u8def\u5f84:</strong> ' + escapeHtml(node.data('path')) + '</div>';
    }
    const sources = node.data('sources') || [];
    if (sources.length > 0) {
        html += '<div><strong>\u6765\u6e90:</strong></div>';
        html += '<ul style="margin:4px 0;padding-left:16px;">';
        sources.forEach(s => {
            html += '<li>' + escapeHtml(s) + '</li>';
        });
        html += '</ul>';
    }

    contentEl.innerHTML = html;

    if (mousePos) {
        let left = mousePos.x + 20;
        let top = mousePos.y - 10;
        if (left + 320 > window.innerWidth) left = mousePos.x - 340;
        if (top + 240 > window.innerHeight) top = window.innerHeight - 260;
        if (top < 0) top = 10;
        popup.style.left = left + 'px';
        popup.style.top = top + 'px';
    } else {
        popup.style.left = '50%';
        popup.style.top = '50%';
        popup.style.transform = 'translate(-50%, -50%)';
    }
    popup.style.display = 'block';
}

function closeNodePopup() {
    const popup = document.getElementById('nodePopup');
    if (popup) {
        popup.style.display = 'none';
        popup.style.transform = '';
    }
}

// ============================================
// Draggable Windows / 可拖动窗口
// ============================================
function makeDraggable(element, handle) {
    let dragState = null;

    handle.addEventListener('mousedown', (e) => {
        const rect = element.getBoundingClientRect();
        dragState = {
            element: element,
            startX: e.clientX,
            startY: e.clientY,
            startLeft: rect.left,
            startTop: rect.top
        };
        element.style.position = 'fixed';
        element.style.left = rect.left + 'px';
        element.style.top = rect.top + 'px';
        element.style.right = 'auto';
        element.style.margin = '0';
        element.style.transform = 'none';
        document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', (e) => {
        if (!dragState) return;
        const dx = e.clientX - dragState.startX;
        const dy = e.clientY - dragState.startY;
        let newLeft = dragState.startLeft + dx;
        let newTop = dragState.startTop + dy;
        const maxLeft = window.innerWidth - dragState.element.offsetWidth;
        const maxTop = window.innerHeight - dragState.element.offsetHeight;
        newLeft = Math.max(0, Math.min(maxLeft, newLeft));
        newTop = Math.max(0, Math.min(maxTop, newTop));
        dragState.element.style.left = newLeft + 'px';
        dragState.element.style.top = newTop + 'px';
    });

    document.addEventListener('mouseup', () => {
        if (dragState) {
            dragState = null;
            document.body.style.userSelect = '';
        }
    });
}

// ============================================
// Global Events / 全局事件
// ============================================
document.addEventListener('click', function(e) {
    const nodePopup = document.getElementById('nodePopup');
    if (nodePopup && nodePopup.style.display === 'block' && !nodePopup.contains(e.target)) {
        const isNode = e.target.closest('.cluster-canvas');
        if (!isNode) closeNodePopup();
    }

    const sourcePopup = document.getElementById('sourcePopup');
    if (sourcePopup && sourcePopup.style.display === 'flex' && !sourcePopup.contains(e.target)) {
        closeSourcePopup();
    }

    const wikiNodeModal = document.getElementById('wikiNodeModal');
    if (wikiNodeModal && wikiNodeModal.style.display === 'flex' && !e.target.closest('.wiki-node-box')) {
        closeWikiNodeModal();
    }
});

// ============================================
// Initialization / 初始化
// ============================================
window.onload = function() {
    initTheme();
    initModeSwitching();
    initLayoutResizers();
    initReferenceBodyResizer();

    document.getElementById('themeToggle').addEventListener('click', toggleTheme);

    document.getElementById('relevanceSlider').addEventListener('input', function() {
        document.getElementById('relevanceValue').textContent = this.value;
    });

    const wikiPanel = document.getElementById('wikiPanel');
    const wikiPanelHeader = wikiPanel ? wikiPanel.querySelector('.wiki-panel-header') : null;
    if (wikiPanel && wikiPanelHeader) {
        makeDraggable(wikiPanel, wikiPanelHeader);
    }

    // Initialize first conversation
    createNewConversation();

    document.getElementById('chatInput').focus();
};
