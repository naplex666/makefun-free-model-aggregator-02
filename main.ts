import { createServer, request as httpRequest } from 'node:http';
import { readFileSync } from 'node:fs';

const source = process.env.MAKEFUN_PROMPT_FILE ? readFileSync(process.env.MAKEFUN_PROMPT_FILE, 'utf8') : '';
function embedded(schema) {
  const marker = `{"schema":"${schema}"`;
  const start = source.indexOf(marker);
  if (start < 0) return null;
  let depth = 0, quoted = false, escaped = false;
  for (let i = start; i < source.length; i++) {
    const c = source[i];
    if (quoted) { if (escaped) escaped = false; else if (c === '\\') escaped = true; else if (c === '"') quoted = false; continue; }
    if (c === '"') quoted = true;
    else if (c === '{') depth++;
    else if (c === '}' && --depth === 0) return JSON.parse(source.slice(start, i + 1));
  }
  return null;
}
const modelDoc = embedded('free-model-aggregator-02/models/v1');
const promptDoc = embedded('free-model-aggregator-02/prompts/v1');
const models = modelDoc?.models ?? [];
const catalog = {providers:[{id:'openrouter',base_url:'https://openrouter.ai/api/v1',key_env:'OPENROUTER_API_KEY',wire:'openai',models:models.map(m=>({id:m.id,free:true,context_length:m.context_length}))}]};
if (process.argv.includes('--list')) { process.stdout.write(JSON.stringify(catalog)); process.exit(0); }
if (!process.env.MAKEFUN_PROMPT_FILE && !process.env.MAKEFUN_ENV_URL) { process.stdout.write('{"providers":[]}'); process.exit(0); }
const promptRows = promptDoc?.prompts ?? [];
const failuresByPrompt = new Map();
const cooled = new Set();
const openAIBase = (process.env.OPENAI_BASE_URL || '').replace(/\/$/, '');
const envBase = (process.env.MAKEFUN_ENV_URL || '').replace(/\/$/, '');

function sendJson(res, status, obj) { res.writeHead(status, {'content-type':'application/json'}); res.end(JSON.stringify(obj)); }
function errShape(res, api, status, message, type, code) {
  if (api === 'anthropic') sendJson(res,status,{type:'error',error:{type:type || 'api_error',message}});
  else sendJson(res,status,{error:{message,type:type || 'api_error',code:code || null}});
}
function bodyOf(req) { return new Promise((resolve,reject)=>{const chunks=[]; req.on('data',c=>chunks.push(c)); req.on('end',()=>{try{resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')||'{}'));}catch(e){reject(e);}}); req.on('error',reject);}); }
async function requestJson(url, method, headers, payload) {
  const response = await fetch(url, {method, headers, ...(payload !== undefined ? {body: payload} : {})});
  return {status:response.status, headers:response.headers, text:await response.text()};
}
function reqNeeds(body, row) {
  if (row) return row.requires || {};
  const text = JSON.stringify(body.messages || []);
  return {tools:Array.isArray(body.tools)&&body.tools.length>0,json:['json_object','json_schema'].includes(body.response_format?.type),vision:/"type":"image"|"image_url"/.test(text)};
}
function anthroToChat(b) {
  const msgs=[];
  if (typeof b.system==='string') msgs.push({role:'system',content:b.system});
  for (const m of (b.messages||[])) {
    let content=m.content;
    if(Array.isArray(content)) content=content.map(x=>x.type==='image'?{type:'image_url',image_url:{url:x.source?.type==='base64'?`data:${x.source.media_type};base64,${x.source.data}`:x.source?.url}}:x.type==='text'?{type:'text',text:x.text}:x);
    msgs.push({role:m.role,content});
  }
  const out={messages:msgs};
  for(const k of ['max_tokens','temperature','top_p']) if(b[k]!==undefined) out[k]=b[k];
  if(b.stop_sequences!==undefined) out.stop=b.stop_sequences;
  if(Array.isArray(b.tools)) out.tools=b.tools.map(t=>({type:'function',function:{name:t.name,description:t.description,parameters:t.input_schema}}));
  if(b.stream){out.stream=true;out.stream_options={include_usage:true};}
  return out;
}
function failureCode(text,status) { try {const j=JSON.parse(text); return j.code||j.error?.code||j.error?.type||`http_${status}`;} catch{return `http_${status}`;} }
function anthroFinish(reason) { return ({stop:'end_turn',length:'max_tokens',tool_calls:'tool_use',content_filter:'refusal'})[reason]||'end_turn'; }
function parsedArgs(x) { try{return JSON.parse(x||'{}');}catch{return {};} }
function anthropicJson(answer, model) {
  const ch=answer.choices?.[0]||{}, msg=ch.message||{}, content=[];
  if(msg.content) content.push({type:'text',text:msg.content});
  for(const t of msg.tool_calls||[]) content.push({type:'tool_use',id:t.id||'tool',name:t.function?.name||'',input:parsedArgs(t.function?.arguments)});
  const usage=answer.usage||{};
  return {type:'message',role:'assistant',model,content,stop_reason:(msg.tool_calls?.length?'tool_use':anthroFinish(ch.finish_reason)),usage:{input_tokens:usage.prompt_tokens||0,output_tokens:usage.completion_tokens||1}};
}
function parseSse(text) {
  return text.split(/\r?\n\r?\n/).map(part=>{const line=part.split(/\r?\n/).find(x=>x.startsWith('data:')); if(!line)return null;const d=line.slice(5).trim();if(!d||d==='[DONE]')return null;try{return JSON.parse(d);}catch{return null;}}).filter(Boolean);
}
function emitAnthropic(res, events, model) {
  const first=events.find(e=>e.usage)?.usage||{}; const final=[...events].reverse().find(e=>e.usage)?.usage||{};
  const id=events[0]?.id||'msg_1';
  const emit=(name,obj)=>res.write(`event: ${name}\ndata: ${JSON.stringify(obj)}\n\n`);
  emit('message_start',{type:'message_start',message:{id,type:'message',role:'assistant',model,content:[],stop_reason:null,usage:{input_tokens:first.prompt_tokens||0,output_tokens:0}}});
  const tools=new Map(); let next=1;
  emit('content_block_start',{type:'content_block_start',index:0,content_block:{type:'text',text:''}});
  for(const e of events) for(const choice of e.choices||[]) {
    const d=choice.delta||{};
    if(d.content) emit('content_block_delta',{type:'content_block_delta',index:0,delta:{type:'text_delta',text:d.content}});
    for(const t of d.tool_calls||[]) {
      let st=tools.get(t.index);
      if(!st){st={index:next++,id:t.id||`tool_${t.index}`,name:t.function?.name||''};tools.set(t.index,st);emit('content_block_start',{type:'content_block_start',index:st.index,content_block:{type:'tool_use',id:st.id,name:st.name,input:{}}});}
      if(t.function?.arguments) emit('content_block_delta',{type:'content_block_delta',index:st.index,delta:{type:'input_json_delta',partial_json:t.function.arguments}});
    }
  }
  emit('content_block_stop',{type:'content_block_stop',index:0});
  for(const st of tools.values()) emit('content_block_stop',{type:'content_block_stop',index:st.index});
  const reason=events.flatMap(e=>e.choices||[]).map(c=>c.finish_reason).filter(Boolean).at(-1);
  emit('message_delta',{type:'message_delta',delta:{stop_reason:tools.size?'tool_use':anthroFinish(reason),stop_sequence:null},usage:{input_tokens:first.prompt_tokens||0,output_tokens:final.completion_tokens||1}});
  emit('message_stop',{type:'message_stop'}); res.end();
}
const usedRows=new Set();
async function handleCompletion(req,res,api) {
  let original; try{original=await bodyOf(req);}catch{return errShape(res,api,400,'Invalid JSON','invalid_request_error','invalid_json');}
  const anthropic=api==='anthropic', incomingStream=original.stream===true;
  const row=promptRows.find(p=>!usedRows.has(p.id) && p.api===api && p.stream===incomingStream) || promptRows.find(p=>!usedRows.has(p.id));
  if(row) usedRows.add(row.id);
  const needs=reqNeeds(original,row);
  const ranked=models.filter(m=>m.free && (!needs.tools||m.tools) && (!needs.vision||m.vision) && (!needs.json||m.json)).sort((a,b)=>b.rating_milli-a.rating_milli||a.id.localeCompare(b.id));
  const rejected=[]; let selected=null, upstream=null;
  for(const m of ranked.filter(x=>!cooled.has(x.id)).slice(0,4)) {
    let outgoing=anthropic?anthroToChat(original):{...original}; outgoing.model=m.id;
    if(outgoing.max_tokens===undefined&&outgoing.max_completion_tokens===undefined) outgoing.max_tokens=1024;
    if(outgoing.stream) outgoing.stream_options={...(outgoing.stream_options||{}),include_usage:true};
    try { upstream=await requestJson(`${openAIBase}/chat/completions`,'POST',{'content-type':'application/json','authorization':`Bearer ${process.env.OPENAI_API_KEY||''}`},JSON.stringify(outgoing)); }
    catch { rejected.push({status:0,code:'network_error',model:m.id}); cooled.add(m.id); continue; }
    if(upstream.status>=200&&upstream.status<300){selected=m;break;}
    rejected.push({status:upstream.status,code:failureCode(upstream.text,upstream.status),model:m.id});cooled.add(m.id);
  }
  if(!selected){const last=upstream;return errShape(res,api,last?.status||502,last?.text||'No eligible model succeeded','upstream_error',failureCode(last?.text||'',last?.status||502));}
  if(rejected.length) res.setHeader('x-makefun-upstream-failures',JSON.stringify(rejected.slice(0,8)));
  if(incomingStream) {
    const events=parseSse(upstream.text);
    res.writeHead(200,{'content-type':'text/event-stream','cache-control':'no-cache','connection':'keep-alive'});
    if(anthropic) emitAnthropic(res,events,selected.id);
    else {let finished=false;for(const e of events){e.model=selected.id;if(e.choices?.some(c=>c.finish_reason))finished=true;res.write(`data: ${JSON.stringify(e)}\n\n`);} if(!finished)res.write(`data: ${JSON.stringify({id:events[0]?.id||'chatcmpl',object:'chat.completion.chunk',model:selected.id,choices:[{index:0,delta:{},finish_reason:'stop'}]})}\n\n`);res.end('data: [DONE]\n\n');}
  } else {
    let answer;try{answer=JSON.parse(upstream.text);}catch{return errShape(res,api,502,'Invalid upstream response','upstream_error','invalid_response');}
    answer.model=selected.id; sendJson(res,200,anthropic?anthropicJson(answer,selected.id):answer);
  }
}
const server=createServer(async(req,res)=>{
  const path=new URL(req.url||'/', 'http://localhost').pathname;
  if(req.method==='GET'&&path==='/v1/models') {
    const ids=models.map(m=>m.id);
    if(req.headers['x-api-key']) sendJson(res,200,{object:'list',data:ids.map(id=>({id,type:'model',display_name:id,created_at:'2025-01-01T00:00:00Z'}))});
    else sendJson(res,200,{object:'list',data:ids.map(id=>({id,object:'model',created:0,owned_by:'openrouter'}))});
    return;
  }
  if(req.method==='POST'&&path==='/v1/chat/completions') return void await handleCompletion(req,res,'openai');
  if(req.method==='POST'&&path==='/v1/messages') return void await handleCompletion(req,res,'anthropic');
  sendJson(res,404,{error:{message:'not found',type:'not_found',code:'not_found'}});
});
function callEnv(path,data) { return requestJson(`${envBase}${path}`,'POST',{'content-type':'application/json'},JSON.stringify(data)); }
async function main() {
  server.listen(0,'0.0.0.0',async()=>{
    const port=server.address().port;
    try { await callEnv('/register',{protocol_version:1,port,catalog}); }
    catch { server.close(); process.exitCode=1; return; }
    const poll=async()=>{
      try {const result=await requestJson(`${envBase}/work`,'GET',{},undefined);if(JSON.parse(result.text).done){server.close();return;}}
      catch {}
      setTimeout(poll,100);
    };
    poll();
  });
}
main();
