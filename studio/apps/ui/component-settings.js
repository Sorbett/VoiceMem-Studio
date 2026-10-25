/* Shared component configuration for both Studio styles. */
((root, factory) => {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.VMComponents = api;
})(globalThis, () => {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  const groups = [
    {id:'input', title:'语音输入', description:'流式识别与最终转写', roles:['asr','asr_final']},
    {id:'memory', title:'记忆系统', description:'记忆检索与整理', roles:['memory']},
    {id:'reply', title:'回复模型', description:'流式模型回复', roles:['reply']},
    {id:'speech', title:'语音合成', description:'回复与附和共用音色', roles:['speech']},
    {id:'turn', title:'轮次控制', description:'语音检测、结束判断与思考路由', roles:['vad','eot','router']},
    {id:'pet', title:'桌宠', description:'跟随回复状态', roles:[]},
  ];

  function comparable(value) {
    if (Array.isArray(value)) return value.map(comparable);
    if (!value || typeof value !== 'object') return value;
    if (Object.hasOwn(value,'optionsText')) {
      value = {...value};
      try { value.options = JSON.parse(value.optionsText || '{}'); delete value.optionsText; } catch {}
    }
    return Object.fromEntries(Object.keys(value).sort().filter(key =>
      !['configured','hasKey'].includes(key) && !(key === 'apiKey' && !value[key]) && value[key] !== ''
    ).map(key => [key, comparable(value[key])]));
  }

  class Drafts {
    constructor(services = {}) { this.services = services; this.changes = new Map(); }
    get size() { return this.changes.size; }
    get(role) { return this.changes.get(role) || this.services[role] || {provider:'recommended'}; }
    set(role, value) {
      if (JSON.stringify(comparable(value)) === JSON.stringify(comparable(this.services[role] || {provider:'recommended'}))) this.changes.delete(role);
      else this.changes.set(role, {...value});
    }
    clear() { this.changes.clear(); }
    batch() {
      return Object.fromEntries([...this.changes].map(([role, value]) => {
        const {optionsText, ...config} = value;
        if (optionsText !== undefined) {
          try { config.options = JSON.parse(optionsText || '{}'); }
          catch { throw new Error('适配器参数必须为有效的 JSON 对象。'); }
          if (!config.options || typeof config.options !== 'object' || Array.isArray(config.options)) throw new Error('适配器参数必须为有效的 JSON 对象。');
        }
        return [role, config];
      }));
    }
  }

  function mount(container, {desktop, defaults, providerName, icons, t = value => value, translate = () => {}} = {}) {
    let runtime = {}, manager = {managed:false}, catalog = {}, drafts = new Drafts();
    let selected = '', activeRole = '', refreshTask, loaded = false, busy = false;
    container.innerHTML = `<p class="component-load" role="status">读取中…</p><div class="component-list"></div>
      <section class="component-editor" aria-labelledby="componentTitle" hidden>
        <header><button type="button" class="component-editor-close" aria-label="返回组件"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m14 6-6 6 6 6"/></svg></button><h3 id="componentTitle"></h3></header>
        <div class="component-subtabs" role="group"></div><div class="component-editor-body"></div>
      </section>
      <div class="component-apply" hidden><p class="component-service-status component-global-status" role="status" aria-live="polite"></p><span class="component-change-count" role="status"></span><button type="button" class="component-discard">撤销更改</button><button type="button" class="component-service-save">应用并重启</button></div>`;
    const list = container.querySelector('.component-list'), editor = container.querySelector('.component-editor');
    const body = editor.querySelector('.component-editor-body'), tabs = editor.querySelector('.component-subtabs');
    const applyBar = container.querySelector('.component-apply'), applyButton = applyBar.querySelector('.component-service-save');
    const status = container.querySelector('.component-global-status'), loadStatus = container.querySelector('.component-load');
    function label(role, service = drafts.get(role)) {
      return catalog[role]?.providers?.[service.provider]?.label || providerName(service.provider);
    }
    function renderList() {
      list.innerHTML = groups.map(group => {
        const changed = group.roles.some(role => drafts.changes.has(role));
        const summary = changed ? '待应用' : group.roles.length ? label(group.roles[0]) : 'App';
        return `<button type="button" class="component-open" data-component="${group.id}"><span class="component-icon"><svg viewBox="0 0 24 24" aria-hidden="true">${icons[group.id] || icons.reply}</svg></span><span class="component-copy"><strong>${group.title}</strong><small>${group.description}</small></span><span class="component-state${changed?' pending':''}">${escape(summary)}</span><svg class="component-chevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7"/></svg></button>`;
      }).join('');
      for (const node of list.querySelectorAll('.component-open')) node.onclick = () => open(node.dataset.component);
      translate(container);
    }
    function updateApply() {
      applyBar.hidden = !manager.managed || (!drafts.size && !busy);
      applyButton.disabled = !drafts.size || busy || manager.reconfiguring;
      applyBar.querySelector('.component-discard').disabled = busy;
      applyBar.querySelector('.component-change-count').textContent = busy ? t('正在重启…') : `${drafts.size} ${t('项待应用')}`;
    }
    function field(name, labelText, value, {required = false, type = 'text', placeholder = ''} = {}) {
      return `<label>${labelText}<input name="${name}" type="${type}" value="${escape(value)}" ${required?'required':''} maxlength="${name==='apiKey'?8192:1000}" spellcheck="false" ${name==='apiKey'?'autocomplete="new-password"':''} placeholder="${escape(placeholder)}"></label>`;
    }
    function retainedKey(role, value) {
      const current = drafts.services[role] || {};
      return (current.hasKey ?? current.configured) && value.provider===current.provider
        && (value.baseUrl||'').replace(/\/$/,'')===(current.baseUrl||'').replace(/\/$/,'')
        && (value.factory||'')===(current.factory||'');
    }
    function capture() {
      const form = body.querySelector('form');
      if (editor.hidden || !activeRole || !form || !manager.managed || busy) return;
      const value = {provider:form.elements.provider.value};
      for (const name of ['model','baseUrl','apiKey','voice','language','factory','optionsText']) {
        if (form.elements[name]) value[name] = form.elements[name].value;
      }
      drafts.set(activeRole, value);
      if (form.elements.apiKey) {
        form.elements.apiKey.placeholder = t(retainedKey(activeRole,value)?'已保存，留空保留':value.provider==='adapter'?'可选':'请输入 API Key');
      }
      status.textContent = ''; renderList(); updateApply();
    }
    function renderForm() {
      const role = activeRole, service = drafts.get(role), legacy = role === 'memory' || role === 'reply';
      const providers = legacy ? {deepseek:{label:'DeepSeek · 推荐'},qwen:{label:'Qwen / DashScope'},openai:{label:'OpenAI-compatible'},...(role==='reply'&&runtime.backend==='mlx'?{local:{label:'本地模型 / MLX'}}:{})} : catalog[role]?.providers;
      if (!providers) { body.innerHTML = '<p class="component-reconfigure-note">此服务版本不支持组件配置。</p>'; return; }
      const choices = Object.entries(providers).filter(([id]) => id !== 'adapter');
      const advanced = providers.adapter ? `<optgroup label="${t('高级')}"><option value="adapter"${service.provider==='adapter'?' selected':''}>自定义适配器</option></optgroup>` : '';
      let fields = '';
      const recommended = service.provider === 'recommended', local = service.provider === 'local', adapter = service.provider === 'adapter';
      if (!recommended) {
        if (adapter) fields += field('factory','Python 适配器',service.factory,{required:true,placeholder:'my_adapter:create'});
        fields += field('model',local?'本地模型路径':'模型名称',service.model,{required:!adapter,placeholder:local?'模型目录或 Hugging Face ID':''});
        if (!local) {
          fields += field('baseUrl','服务地址',service.baseUrl,{required:!adapter,type:'url'});
          fields += field('apiKey','API Key','',{type:'password',placeholder:retainedKey(role,service)?'已保存，留空保留':adapter?'可选':'请输入 API Key'});
        }
        if (service.provider==='openai'&&role==='speech') fields += field('voice','音色 ID',service.voice,{required:true});
        if (service.provider==='deepgram') fields += field('language','识别语言',service.language,{required:true,placeholder:'zh / en / multi'});
        if (adapter) fields += `<label>适配器参数<textarea name="optionsText" spellcheck="false" rows="4" maxlength="16384">${escape(service.optionsText ?? JSON.stringify(service.options || {},null,2))}</textarea></label>`;
      }
      const contract = catalog[role]?.contract || (role==='memory'?'结构化记忆提取 · OpenAI Chat 协议':'流式文本 · 对话历史 · 可取消');
      body.innerHTML = `<p class="component-contract">${escape(contract)}</p><form class="component-service-form" data-role="${role}"><fieldset${!manager.managed||busy?' disabled':''}><label>接入方式<select name="provider">${choices.map(([value,spec])=>`<option value="${value}"${value===service.provider?' selected':''}>${escape(spec.label)}</option>`).join('')}${advanced}</select></label>${fields}</fieldset></form>
        ${recommended?'<div class="component-recommended"><span aria-hidden="true">✓</span><span>使用推荐链路</span></div>':''}
        ${role==='speech'&&!recommended?'<p class="component-reconfigure-note">首次启用此音色会合成 2 段附和音频。</p>':''}
        ${manager.managed?`<button type="button" class="component-reset"${busy?' disabled':''}>恢复推荐配置</button>`:'<p class="component-reconfigure-note">此服务由外部管理，请在服务器端修改配置。</p>'}`;
      const form = body.querySelector('form');
      // Draft secrets live only in this settings instance, never in browser storage.
      if (form.elements.apiKey) form.elements.apiKey.value = service.apiKey || '';
      form.oninput = event => { if (event.target.name !== 'provider') capture(); };
      form.onsubmit = event => { event.preventDefault(); void apply(); };
      form.elements.provider.onchange = () => {
        const provider = form.elements.provider.value;
        const value = legacy ? {...defaults[provider],provider} : {provider,...(providers[provider] || {})};
        delete value.label;
        drafts.set(role, value); status.textContent = ''; renderForm(); renderList(); updateApply();
      };
      body.querySelector('.component-reset')?.addEventListener('click', () => {
        if (busy) return;
        drafts.set(role, legacy ? {provider:'deepseek',...defaults.deepseek} : {provider:'recommended'});
        renderForm(); renderList(); updateApply();
      });
      translate(body);
    }
    function open(id, role) {
      const switching = selected===id && !!role;
      capture(); selected = id;
      const group = groups.find(item => item.id===id);
      editor.querySelector('h3').textContent = t(group.title);
      activeRole = role || group.roles[0] || '';
      tabs.hidden = group.roles.length<2;
      tabs.setAttribute('aria-label',t(group.title));
      tabs.innerHTML = group.roles.map(value=>`<button type="button" data-role="${value}" aria-pressed="${value===activeRole}">${escape(catalog[value]?.title || value)}</button>`).join('');
      for (const button of tabs.querySelectorAll('button')) button.onclick = () => open(id,button.dataset.role);
      if (id==='pet') body.innerHTML = '<div class="component-field"><span>管理方式</span><strong>App 菜单</strong></div>';
      else renderForm();
      editor.hidden = false; container.classList.add('editing');
      (switching?tabs.querySelector(`[data-role="${activeRole}"]`):editor.querySelector('.component-editor-close'))?.focus(); translate(editor);
    }
    editor.querySelector('.component-editor-close').onclick = () => {
      capture(); editor.hidden = true; container.classList.remove('editing');
      list.querySelector(`[data-component="${selected}"]`)?.focus(); selected = ''; activeRole = '';
    };
    applyBar.querySelector('.component-discard').onclick = () => {
      if (busy) return;
      drafts.clear(); status.textContent = ''; renderList(); updateApply(); if (activeRole) renderForm();
    };
    async function apply() {
      if (!manager.managed || busy || !drafts.size) return;
      const form = body.querySelector('form');
      if (!editor.hidden && form && !form.reportValidity()) return;
      try {
        const updates = drafts.batch();
        for (const config of Object.values(updates)) {
          if (!['local','recommended','adapter'].includes(config.provider) && (!config.model?.trim() || !config.baseUrl?.trim())) throw new Error('请填写模型名称和服务地址。');
        }
        busy = true; status.textContent = t('正在保存并重启服务…'); updateApply();
        if (activeRole) renderForm();
        if (!desktop.apply) throw new Error('请重启桌面 App 后再配置组件。');
        const result = await desktop.apply(updates);
        if (result?.services) drafts.services = {...drafts.services,...result.services};
        drafts.clear(); renderList(); if (activeRole) renderForm();
        status.textContent = t('配置已提交，正在等待服务重新就绪。'); updateApply();
      } catch (error) {
        busy = false; status.textContent = error.message; updateApply(); if (activeRole) renderForm();
      }
    }
    applyButton.onclick = () => { capture(); void apply(); };
    async function refresh() {
      if (refreshTask || busy) return refreshTask;
      refreshTask = (async () => {
        try {
          const response = await fetch('/api/components',{cache:'no-store'});
          if (!response.ok) throw new Error(t('读取失败，请重试。'));
          runtime = await response.json(); catalog = runtime.catalog || {};
          manager = desktop ? await desktop.state() : {managed:false};
          const services = {...runtime.bindings,...manager.services};
          for (const role of ['memory','reply']) services[role] ||= {...runtime[role],baseUrl:runtime[role]?.base_url || ''};
          drafts.services = services;
          loaded = true; loadStatus.hidden = true; renderList(); updateApply();
          if (selected) open(selected,activeRole);
        } catch (error) {
          loadStatus.hidden = false; loadStatus.textContent = error.message;
          if (!loaded) {
            const retry = document.createElement('button'); retry.type='button'; retry.textContent=t('重试'); retry.onclick=()=>void refresh(); loadStatus.append(' ',retry);
          }
        } finally { refreshTask = null; }
      })();
      return refreshTask;
    }
    return {activate:refresh};
  }
  return {Drafts,mount};
});
