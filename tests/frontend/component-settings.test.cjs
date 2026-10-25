'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const r = require('../../studio/apps/runtime.cjs');
const {Drafts} = require('../../studio/apps/ui/component-settings.js');

function current() {
  return r.modelServices({memory:{provider:'deepseek',apiKey:'memory-fixture'},reply:{provider:'openai',apiKey:'reply-fixture'}});
}

test('existing saved configurations retain the recommended audio and routing factories', () => {
  const config = current();
  for (const role of Object.keys(r.COMPONENT_CATALOG)) assert.deepEqual(config[role],{provider:'recommended'});
  assert.deepEqual(r.modelService('reply',{provider:'local',model:'/fixture/local-model'}),{
    provider:'local',model:'/fixture/local-model',baseUrl:'',apiKey:'',
  });
});

test('component configuration validates protocol requirements and advanced options', () => {
  assert.equal(r.componentService('speech',{provider:'openai'}).voice,'alloy');
  assert.equal(r.componentService('asr',{provider:'deepgram'}).language,'zh');
  assert.throws(()=>r.componentService('speech',{provider:'unknown'}),/接入方式/);
  assert.throws(()=>r.componentService('asr',{provider:'adapter',factory:'/tmp/script.py'}),/适配器/);
  for (const options of [[],{api_key:'fixture'}, {nested:{Authorization:'fixture'}}, {temperature:NaN}]) {
    assert.throws(()=>r.componentService('asr',{provider:'adapter',factory:'example:create',options}));
  }
  const adapter = r.componentService('router',{provider:'adapter',factory:'example.router:create',options:{max_tokens:40}});
  assert.deepEqual(adapter.options,{max_tokens:40});
  assert.throws(()=>r.componentService('vad',{provider:'recommended',unused:true}),/未知/);
});

test('a batch validates all components without mutating the running configuration', () => {
  const before = current(), snapshot = JSON.stringify(before);
  const next = r.updateModelServices(before,{
    asr:{provider:'deepgram',apiKey:'asr-fixture'},
    speech:{provider:'openai',apiKey:'speech-fixture'},
  });
  assert.equal(next.asr.model,'nova-3');
  assert.equal(next.speech.voice,'alloy');
  assert.equal(JSON.stringify(before),snapshot);
  assert.throws(()=>r.updateModelServices(before,{speech:{provider:'openai',apiKey:'speech-fixture'},vad:{provider:'invalid'}}));
  assert.equal(JSON.stringify(before),snapshot);
  assert.throws(()=>r.updateModelServices(before,{unknown:{provider:'recommended'}}));
  assert.throws(()=>r.updateModelServices(before,{}),/没有/);
});

test('retained keys are scoped to the provider, endpoint and adapter factory', () => {
  const before = current();
  assert.equal(r.updateModelService(before,'reply',{provider:'openai',model:'other-model'}).reply.apiKey,'reply-fixture');
  assert.throws(()=>r.updateModelService(before,'reply',{
    provider:'openai',baseUrl:'https://another.example/v1',apiKey:'',
  }),/API Key/);
  const custom = r.updateModelService(before,'reply',{
    provider:'openai',baseUrl:'https://another.example/v1',apiKey:'custom-fixture',
  });
  assert.throws(()=>r.updateModelService(custom,'reply',{provider:'openai',model:'default-endpoint-model'}),/API Key/);
  const local = r.updateModelService(before,'reply',{provider:'openai',baseUrl:'http://localhost:11434/v1',apiKey:''});
  assert.equal(local.reply.apiKey,'');
  const environment = r.modelServiceEnvironment({OPENAI_API_KEY:'unrelated-secret'},local);
  assert.equal(environment.VOICEMEM_STUDIO_API_KEY,'not-required');
  const first = r.updateModelService(before,'asr',{provider:'adapter',factory:'one:create',apiKey:'one-fixture'});
  const second = r.updateModelService(first,'asr',{provider:'adapter',factory:'two:create'});
  assert.equal(second.asr.apiKey,'');
});

test('new credentials are encrypted on disk and absent from public configuration and config JSON', async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(),'studio-components-'));
  t.after(()=>fs.rm(directory,{recursive:true,force:true}));
  const file = path.join(directory,'services.json');
  const config = r.updateModelServices(current(),{
    asr:{provider:'deepgram',apiKey:'asr-fixture'},speech:{provider:'openai',apiKey:'speech-fixture'},
  });
  await r.saveModelServices(file,config,value=>Buffer.from(value).toString('base64'));
  const stored = await fs.readFile(file,'utf8');
  assert.equal(stored.includes('asr-fixture')||stored.includes('speech-fixture'),false);
  assert.deepEqual(await r.loadModelServices(file,value=>Buffer.from(value,'base64').toString()),config);
  assert.equal(JSON.stringify(r.publicModelServices(config)).includes('-fixture'),false);
  assert.equal(r.publicModelServices(config).asr.hasKey,true);
  const env = r.modelServiceEnvironment({STUDIO_ASR_API_KEY:'old-fixture'},config);
  assert.equal(env.STUDIO_ASR_API_KEY,'asr-fixture');
  assert.equal(env.STUDIO_SPEECH_API_KEY,'speech-fixture');
  assert.equal(env.STUDIO_COMPONENTS.includes('-fixture'),false);
  const restored = r.modelServicesFromLaunch({memoryProvider:'deepseek',replyProvider:'openai'},env);
  assert.deepEqual(restored,config);
  await fs.writeFile(file,JSON.stringify({schema:1,memory:{provider:'deepseek',secret:''},reply:{provider:'local',secret:''}}));
  assert.deepEqual((await r.loadModelServices(file,()=>{throw new Error('empty secret');})).asr,{provider:'recommended'});
});

test('component launch requires credentials outside JSON and respects explicit empty role keys', () => {
  const launch = {memoryProvider:'openai',replyProvider:'openai'};
  assert.throws(()=>r.modelServicesFromLaunch(launch,{
    STUDIO_COMPONENTS:'{"asr":{"provider":"deepgram","apiKey":"fixture"}}',
  }),/STUDIO_<ROLE>_API_KEY/);
  const values = r.modelServicesFromLaunch(launch,{
    OPENAI_API_KEY:'unrelated-fixture',VOICEMEM_STUDIO_API_KEY:'',
  });
  assert.equal(values.reply.apiKey,'');
  assert.equal(values.memory.apiKey,'unrelated-fixture');
});

test('drafts preserve independent fields across navigation and discard without a restart', () => {
  const drafts = new Drafts(r.publicModelServices(current()));
  drafts.set('reply',{provider:'openai',model:'custom-chat',baseUrl:'https://api.openai.com/v1',apiKey:''});
  drafts.set('speech',{provider:'openai',model:'custom-speech',baseUrl:'https://speech.example/v1',voice:'voice-id',apiKey:'speech-fixture'});
  assert.equal(drafts.size,2);
  assert.equal(drafts.get('reply').model,'custom-chat');
  assert.equal(drafts.batch().speech.apiKey,'speech-fixture');
  drafts.clear();
  assert.equal(drafts.size,0);
  assert.equal(drafts.get('reply').model,'gpt-4o');
});

test('unchanged masked keys and formatted JSON do not mark a component dirty', () => {
  const services = r.publicModelServices(current());
  services.asr = {provider:'adapter',factory:'example:create',options:{max_tokens:40},configured:true};
  const drafts = new Drafts(services);
  drafts.set('reply',{...services.reply,apiKey:''});
  drafts.set('asr',{provider:'adapter',factory:'example:create',optionsText:'{ "max_tokens": 40 }',apiKey:''});
  assert.equal(drafts.size,0);
  drafts.set('asr',{provider:'adapter',factory:'example:create',optionsText:'[1,2]'});
  assert.throws(()=>drafts.batch(),/JSON 对象/);
  drafts.set('asr',{provider:'recommended'});
  assert.deepEqual(drafts.batch(),{asr:{provider:'recommended'}});
});
