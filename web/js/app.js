/**
 * 多模态RAG系统 - 前端交互逻辑
 */

// ============================================
// 全局配置
// ============================================
const CONFIG = {
    API_BASE_URL: localStorage.getItem('apiBaseUrl') || 'http://localhost:8000',
    DEFAULT_METHOD: localStorage.getItem('defaultMethod') || 'balanced',
    DEFAULT_TOP_K: parseInt(localStorage.getItem('defaultTopK')) || 5
};

// 当前状态
let currentMethod = CONFIG.DEFAULT_METHOD;
let isProcessing = false;

// ============================================
// 初始化
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initChat();
    initSettings();
    checkHealth();
    loadIndexStatus(); // 加载索引状态
});

// ============================================
// 导航功能
// ============================================
function initNavigation() {
    const navItems = document.querySelectorAll('.sidebar-nav li[data-tab]');
    const contentAreas = document.querySelectorAll('.content-area');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabName = item.dataset.tab;
            
            // 更新导航状态
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // 切换内容区域
            contentAreas.forEach(area => area.classList.remove('active'));
            document.getElementById(`${tabName}-tab`).classList.add('active');
            
            // 更新面包屑
            updateBreadcrumb(item.querySelector('span').textContent);
        });
    });
    
    // RAG方案选择
    const methodItems = document.querySelectorAll('.method-item');
    methodItems.forEach(item => {
        item.addEventListener('click', () => {
            methodItems.forEach(m => m.classList.remove('active'));
            item.classList.add('active');
            currentMethod = item.dataset.method;
            updateCurrentMethodDisplay();
        });
    });
}

function updateBreadcrumb(text) {
    document.querySelector('.current-page').textContent = text;
}

// ============================================
// 索引状态管理
// ============================================
async function loadIndexStatus() {
    // 页面加载时获取索引状态
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/index/status`);
        if (!response.ok) return;

        const data = await response.json();

        data.indices.forEach(index => {
            if (index.exists && index.stats) {
                updateStats(index.method, index.stats);
            }
        });
    } catch (error) {
        console.log('加载索引状态失败:', error);
    }
}

function updateCurrentMethodDisplay() {
    const methodNames = {
        'multimodal_vector': '多模态向量模型',
        'multimodal_llm': '多模态大模型',
        'balanced': '平衡方案'
    };
    
    const methodIcons = {
        'multimodal_vector': 'fa-vector-square',
        'multimodal_llm': 'fa-brain',
        'balanced': 'fa-balance-scale'
    };
    
    const display = document.querySelector('.current-method');
    if (display) {
        display.innerHTML = `<i class="fas ${methodIcons[currentMethod]}"></i> 当前方案: ${methodNames[currentMethod]}`;
    }
}

// ============================================
// 聊天功能
// ============================================
function initChat() {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    
    // 自动调整文本框高度
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
    });
    
    // 发送消息
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    sendBtn.addEventListener('click', sendMessage);
}

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const query = messageInput.value.trim();
    
    if (!query || isProcessing) return;
    
    // 添加用户消息
    addMessage('user', query);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // 隐藏欢迎消息
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }
    
    // 显示加载状态
    showLoading('正在思考...');
    isProcessing = true;
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                method: currentMethod,
                top_k: CONFIG.DEFAULT_TOP_K
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '请求失败');
        }
        
        const data = await response.json();
        addMessage('assistant', data.answer, data.sources);
    } catch (error) {
        showToast(error.message, 'error');
        addMessage('assistant', `抱歉，发生了错误：${error.message}`);
    } finally {
        hideLoading();
        isProcessing = false;
    }
}

function sendQuickMessage(message) {
    const messageInput = document.getElementById('messageInput');
    messageInput.value = message;
    sendMessage();
}

function addMessage(role, content, sources = []) {
    const chatMessages = document.getElementById('chatMessages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // 处理Markdown格式
    messageContent.innerHTML = formatMarkdown(content);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    // 添加来源信息
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        
        // 创建可展开的来源列表
        const sourcesHeader = document.createElement('div');
        sourcesHeader.className = 'sources-header';
        sourcesHeader.style.cssText = 'display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 8px 0; color: var(--text-muted); font-size: 0.85rem;';
        sourcesHeader.innerHTML = `
            <i class="fas fa-chevron-right" style="transition: transform 0.2s;"></i>
            <i class="fas fa-link"></i>
            <span>参考来源: ${sources.length} 个文档</span>
        `;
        
        const sourcesList = document.createElement('div');
        sourcesList.className = 'sources-list';
        sourcesList.style.cssText = 'display: none; margin-top: 8px; padding-left: 24px;';
        
        sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            sourceItem.style.cssText = 'padding: 8px 12px; margin-bottom: 6px; background: var(--bg-primary); border-radius: 6px; border-left: 3px solid var(--primary-color);';
            
            const typeIcon = source.type === 'image' ? 'fa-image' : source.type === 'table' ? 'fa-table' : 'fa-file-alt';
            const typeColor = source.type === 'image' ? '#10b981' : source.type === 'table' ? '#f59e0b' : '#6366f1';
            
            sourceItem.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                    <i class="fas ${typeIcon}" style="color: ${typeColor};"></i>
                    <span style="font-weight: 500; color: var(--text-primary);">来源 ${index + 1}</span>
                    <span style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">${source.type}</span>
                </div>
                ${source.content ? `<div style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5; max-height: 100px; overflow-y: auto;">${formatMarkdown(source.content)}</div>` : ''}
            `;
            
            sourcesList.appendChild(sourceItem);
        });
        
        // 点击展开/收起
        let isExpanded = false;
        sourcesHeader.addEventListener('click', () => {
            isExpanded = !isExpanded;
            const icon = sourcesHeader.querySelector('.fa-chevron-right');
            icon.style.transform = isExpanded ? 'rotate(90deg)' : 'rotate(0deg)';
            sourcesList.style.display = isExpanded ? 'block' : 'none';
        });
        
        sourcesDiv.appendChild(sourcesHeader);
        sourcesDiv.appendChild(sourcesList);
        messageContent.appendChild(sourcesDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatMarkdown(text) {
    // 简单的Markdown格式化
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

// ============================================
// 文件上传功能
// ============================================
let selectedFile = null;
let currentPdfFilename = null;

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.name.endsWith('.pdf')) {
        showToast('请选择PDF文件', 'error');
        return;
    }
    
    selectedFile = file;
    document.getElementById('selectedFileName').textContent = file.name;
    document.getElementById('selectedFileName').style.color = 'var(--text-primary)';
    
    showToast(`已选择文件: ${file.name}`, 'success');
}

async function uploadAndBuildIndex(method) {
    if (!selectedFile) {
        showToast('请先选择PDF文件', 'warning');
        document.getElementById('pdfFileInput').click();
        return;
    }
    
    showLoading(`正在上传并构建 ${getMethodName(method)} 索引...`);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('method', method);
    formData.append('clear_existing', 'true');
    
    // 显示上传进度
    const progressDiv = document.getElementById('uploadProgress');
    const progressBar = document.getElementById('uploadProgressBar');
    const percentText = document.getElementById('uploadPercent');
    progressDiv.style.display = 'block';
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/index/upload`, {
            method: 'POST',
            body: formData
        });
        
        // 模拟进度
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 10;
            if (progress <= 90) {
                progressBar.style.width = `${progress}%`;
                percentText.textContent = `${progress}%`;
            }
        }, 200);
        
        if (!response.ok) {
            clearInterval(progressInterval);
            const error = await response.json();
            throw new Error(error.detail || '上传失败');
        }
        
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        percentText.textContent = '100%';
        
        const data = await response.json();
        updateStats(method, data.stats);
        
        // 显示当前PDF信息
        currentPdfFilename = selectedFile.name;
        document.getElementById('currentPdfName').textContent = currentPdfFilename;
        document.getElementById('currentPdfInfo').style.display = 'block';
        
        showToast('文件上传并索引构建成功！', 'success');
        
        // 3秒后隐藏进度条
        setTimeout(() => {
            progressDiv.style.display = 'none';
            progressBar.style.width = '0%';
        }, 3000);
        
    } catch (error) {
        document.getElementById('uploadProgress').style.display = 'none';
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// ============================================
// 索引管理功能
// ============================================
async function buildIndex(method) {
    // 如果有选择文件，先上传文件
    if (selectedFile) {
        await uploadAndBuildIndex(method);
        return;
    }
    
    // 否则使用默认PDF路径构建索引
    showLoading(`正在构建 ${getMethodName(method)} 索引...`);
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/index`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                method: method,
                clear_existing: true
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '构建失败');
        }
        
        const data = await response.json();
        updateStats(method, data.stats);
        showToast('索引构建成功！', 'success');
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function buildAllIndices() {
    showLoading('正在构建所有索引...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/index/all`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const results = await response.json();
        
        results.forEach(result => {
            if (result.success) {
                updateStats(result.method, result.stats);
            }
        });
        
        const successCount = results.filter(r => r.success).length;
        showToast(`成功构建 ${successCount}/3 个索引`, 'success');
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

function updateStats(method, stats) {
    const statsMap = {
        'multimodal_vector': 'stats-vector',
        'multimodal_llm': 'stats-llm',
        'balanced': 'stats-balanced'
    };

    const container = document.getElementById(statsMap[method]);
    if (!container) return;

    const statItems = container.querySelectorAll('.stat-value');

    // 如果没有统计信息或为空对象，显示 "-"
    if (!stats || Object.keys(stats).length === 0) {
        statItems.forEach(item => {
            item.textContent = '-';
        });
        return;
    }

    const keys = Object.keys(stats);

    statItems.forEach((item, index) => {
        if (keys[index]) {
            item.textContent = stats[keys[index]];
        }
    });
}

async function clearIndex(method) {
    if (!confirm(`确定要清空 ${getMethodName(method)} 的索引吗？`)) return;

    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/index/${method}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '清空失败');
        }

        const data = await response.json();

        if (data.success) {
            // 清空前端显示的统计
            updateStats(method, {});
            showToast(`${getMethodName(method)} 索引已清空`, 'success');
        } else {
            showToast(data.message || '清空失败', 'warning');
        }
    } catch (error) {
        showToast(error.message, 'error');
    }
}

async function clearAllIndices() {
    if (!confirm('确定要清空所有索引吗？')) return;

    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/index`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '清空失败');
        }

        const data = await response.json();

        if (data.success) {
            // 清空所有前端显示的统计
            ['multimodal_vector', 'multimodal_llm', 'balanced'].forEach(method => {
                updateStats(method, {});
            });
            showToast(`已清空 ${data.stats?.cleared || 0}/3 个索引`, 'success');
        } else {
            showToast(data.message || '清空失败', 'warning');
        }
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function getMethodName(method) {
    const names = {
        'multimodal_vector': '多模态向量模型',
        'multimodal_llm': '多模态大模型',
        'balanced': '平衡方案'
    };
    return names[method] || method;
}

// ============================================
// 方案对比功能
// ============================================
async function compareMethods() {
    const queryInput = document.getElementById('compareQuery');
    const query = queryInput.value.trim();
    
    if (!query) {
        showToast('请输入查询问题', 'warning');
        return;
    }
    
    showLoading('正在对比三种方案...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/query/all`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                top_k: CONFIG.DEFAULT_TOP_K
            })
        });
        
        const results = await response.json();
        displayCompareResults(results);
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

function displayCompareResults(results) {
    const container = document.getElementById('compareResults');
    container.innerHTML = '';
    
    const methodConfig = {
        'multimodal_vector': { name: '多模态向量模型', icon: 'fa-vector-square', color: 'blue' },
        'multimodal_llm': { name: '多模态大模型', icon: 'fa-brain', color: 'purple' },
        'balanced': { name: '平衡方案', icon: 'fa-balance-scale', color: 'green' }
    };
    
    results.forEach(result => {
        const config = methodConfig[result.method];
        
        const card = document.createElement('div');
        card.className = 'compare-card';
        card.innerHTML = `
            <div class="compare-card-header">
                <div class="method-icon ${config.color}">
                    <i class="fas ${config.icon}"></i>
                </div>
                <h4>${config.name}</h4>
            </div>
            <div class="compare-card-body">
                <div class="answer">${formatMarkdown(result.answer)}</div>
                ${result.sources ? `<div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color); font-size: 0.8rem; color: var(--text-muted);">
                    <i class="fas fa-link"></i> 参考来源: ${result.sources.length} 个文档
                </div>` : ''}
            </div>
        `;
        
        container.appendChild(card);
    });
}

// ============================================
// 设置功能
// ============================================
function initSettings() {
    const apiBaseUrl = document.getElementById('apiBaseUrl');
    const defaultTopK = document.getElementById('defaultTopK');
    const defaultMethod = document.getElementById('defaultMethod');
    
    if (apiBaseUrl) apiBaseUrl.value = CONFIG.API_BASE_URL;
    if (defaultTopK) defaultTopK.value = CONFIG.DEFAULT_TOP_K;
    if (defaultMethod) defaultMethod.value = CONFIG.DEFAULT_METHOD;
}

function saveSettings() {
    const apiBaseUrl = document.getElementById('apiBaseUrl').value;
    const defaultTopK = document.getElementById('defaultTopK').value;
    const defaultMethod = document.getElementById('defaultMethod').value;
    
    CONFIG.API_BASE_URL = apiBaseUrl;
    CONFIG.DEFAULT_TOP_K = parseInt(defaultTopK);
    CONFIG.DEFAULT_METHOD = defaultMethod;
    
    localStorage.setItem('apiBaseUrl', apiBaseUrl);
    localStorage.setItem('defaultTopK', defaultTopK);
    localStorage.setItem('defaultMethod', defaultMethod);
    
    showToast('设置已保存', 'success');
}

async function testConnection() {
    showLoading('正在测试连接...');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        if (response.ok) {
            showToast('连接成功！', 'success');
        } else {
            throw new Error('服务异常');
        }
    } catch (error) {
        showToast('连接失败：' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// ============================================
// 健康检查
// ============================================
async function checkHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        const data = await response.json();
        
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (data.status === 'healthy') {
            statusDot.classList.add('online');
            statusText.textContent = '服务正常';
        } else {
            statusDot.classList.remove('online');
            statusText.textContent = '服务异常';
        }
    } catch (error) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        statusDot.classList.remove('online');
        statusText.textContent = '无法连接';
    }
}

// ============================================
// 工具函数
// ============================================
function showLoading(text = '加载中...') {
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    loadingText.textContent = text;
    overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.remove('active');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-times-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    toast.innerHTML = `
        <i class="fas ${icons[type]}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
