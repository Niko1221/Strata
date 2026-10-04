// English is the default and the fallback. UI language never changes API values or model prompts.
"use strict";
window.StrataI18n = (() => {
  const ar = {
    "{rate} tok/s": "{rate} رمز/ث", "image": "صورة", "GPU": "بطاقة الرسوميات", "RAM": "ذاكرة النظام", "Tool": "أداة", "s": "ث",
    "Language": "اللغة", "Views": "الصفحات", "Chat": "المحادثة", "Monitor": "المراقبة", "About": "حول",
    "Connecting…": "جارٍ الاتصال…", "Light / dark": "فاتح / داكن",
    "Switch between light and dark": "التبديل بين المظهر الفاتح والداكن", "Switch color theme": "تبديل المظهر",
    "Ask anything": "اسأل ما تريد", "Ask anything…": "اسأل ما تريد…", "Message": "الرسالة",
    "The model runs on this PC. Nothing leaves it.": "يعمل النموذج على هذا الجهاز. لا تغادره بياناتك.",
    "{model} runs on this PC. Nothing leaves it.": "يعمل {model} على هذا الجهاز. لا تغادره بياناتك.",
    "Attach a text file (or drop it here)": "أرفق ملفًا نصيًا (أو أسقطه هنا)",
    "Attach a text file or a picture (or drop it here)": "أرفق ملفًا نصيًا أو صورة (أو أسقطه هنا)",
    "Attach a file": "إرفاق ملف", "New chat": "محادثة جديدة", "Save this chat as Markdown": "حفظ المحادثة بصيغة Markdown",
    "Save this chat": "حفظ المحادثة", "Sampling and thinking": "إعدادات التوليد والتفكير", "Stop": "إيقاف", "Send": "إرسال",
    "Model state": "حالة النموذج", "Idle": "خامل", "Reading": "يقرأ", "Generating": "يولّد", "Queued": "في الانتظار", "Error": "خطأ",
    "Waiting for a request": "بانتظار طلب", "Context fill": "استخدام السياق", "Experts in VRAM": "الخبراء في ذاكرة البطاقة",
    "System RAM": "ذاكرة النظام", "GPU temperature": "حرارة البطاقة", "Recent requests": "الطلبات الأخيرة",
    "Show all": "عرض الكل", "Show fewer": "عرض أقل", "Show all ({count})": "عرض الكل ({count})",
    "Time": "الوقت", "Status": "الحالة", "Prompt": "المدخلات", "Reused": "المُعاد استخدامها", "Output": "المخرجات",
    "Tok/s": "رمز/ث", "Hit rate": "نسبة إصابة الذاكرة", "Duration": "المدة", "No requests yet": "لا توجد طلبات بعد",
    "While writing the answer: the share of the experts looked up that were already in VRAM (experts copied over PCIe are not counted)": "أثناء كتابة الإجابة: نسبة الخبراء الموجودين مسبقًا في ذاكرة البطاقة (لا تشمل المنقولين عبر PCIe)",
    "MCP servers": "خوادم MCP", "Their tools run on this PC with your rights, when the model decides to call them (in this page's chat only; switch it off in Sampling).": "تعمل أدواتها على هذا الجهاز بصلاحياتك عندما يقرر النموذج استدعاءها (في محادثة هذه الصفحة فقط؛ يمكنك تعطيلها من إعدادات التوليد).",
    "Model and engine": "النموذج والمحرك", "This PC": "هذا الجهاز", "Connect your tools": "ربط أدواتك",
    "Any OpenAI- or Anthropic-compatible client works with these addresses.": "يمكن لأي تطبيق متوافق مع OpenAI أو Anthropic استخدام هذه العناوين.",
    "Settings": "الإعدادات", "API key": "مفتاح API", "only if the server was started with one": "إذا شُغّل الخادم بمفتاح فقط",
    "Not needed": "غير مطلوب", "Dark theme": "المظهر الداكن", "Chats, settings and the key are kept in this browser only.": "تُحفظ المحادثات والإعدادات والمفتاح في هذا المتصفح فقط.",
    "Strata on GitHub": "Strata على GitHub", "Sampling": "إعدادات التوليد", "Close": "إغلاق", "Thinking": "التفكير",
    "Off": "معطّل", "Low": "منخفض", "Medium": "متوسط", "High": "مرتفع", "Temperature": "درجة العشوائية",
    "0 = always the most likely word (exact, repeatable)": "0 = اختيار الكلمة الأكثر احتمالًا دائمًا (نتائج قابلة للتكرار)",
    "Top-p": "Top-p", "Top-k": "Top-k", "Max tokens": "الحد الأقصى للرموز", "empty = until done": "فارغ = حتى الاكتمال",
    "Until done": "حتى الاكتمال", "Seed": "بذرة العشوائية", "empty = random": "فارغ = عشوائي", "Random": "عشوائي",
    "Show thinking": "عرض التفكير", "expanded while it streams": "يظهر موسّعًا أثناء التوليد",
    "Use tools from MCP servers": "استخدام أدوات خوادم MCP", "the model may call them while it answers": "قد يستدعيها النموذج أثناء الإجابة",
    "Experimental speed projection": "إسقاط تسريع تجريبي", "the engine's control vector; off = the stock model. Switching reads the chat again once": "متجه التحكم في المحرك؛ تعطيله يستخدم النموذج الأصلي. يؤدي التبديل إلى إعادة قراءة المحادثة مرة واحدة",
    "Use for other apps too": "تطبيقها على التطبيقات الأخرى أيضًا", "omp and other API clients get these settings for anything they don't set themselves": "يستخدم omp وتطبيقات API الأخرى هذه الإعدادات ما لم تحدد إعداداتها الخاصة",
    "Reset": "إعادة الضبط", "Apply": "تطبيق", "Copied to clipboard": "نُسخ إلى الحافظة", "API key saved": "حُفظ مفتاح API",
    "Kept in this browser only.": "يُحفظ في هذا المتصفح فقط.", "API key needed": "مفتاح API مطلوب",
    "This server needs a key: add it under About > Settings.": "يحتاج الخادم إلى مفتاح: أضفه في صفحة حول، ضمن الإعدادات.",
    "This server needs an API key: add it under About > Settings.": "يحتاج الخادم إلى مفتاح API: أضفه في صفحة حول، ضمن الإعدادات.",
    "Server not reachable": "تعذّر الاتصال بالخادم", "Reading prompt": "جارٍ قراءة المدخلات", "Reading prompt · {pct}%": "جارٍ قراءة المدخلات · {pct}%",
    "Generating · {rate} tok/s": "جارٍ التوليد · {rate} رمز/ث", "{count} queued": "{count} في الانتظار",
    "{count} tokens": "{count} رمز", "{read} / {total} tokens · {pct}%": "{read} / {total} رمز · {pct}%",
    "{count} tokens · {rate} tok/s": "{count} رمز · {rate} رمز/ث", " at {rate} tok/s": " بسرعة {rate} رمز/ث",
    "last: {count} tokens{speed}": "الأخير: {count} رمز{speed}", "Thinking…": "جارٍ التفكير…", "Writing": "جارٍ الكتابة",
    "Thinking phase": "التفكير", "Answering": "الإجابة", "Speed": "السرعة", "GPU load": "حمل البطاقة", "VRAM": "ذاكرة البطاقة",
    "GPU temp": "حرارة البطاقة", "Power": "الطاقة", "CPU": "المعالج", "Disk read": "قراءة القرص",
    "Decode": "توليد الرموز", "Prefill": "قراءة المدخلات", "Decode now": "التوليد الآن", "Decode last request": "توليد آخر طلب",
    "Prefill now": "القراءة الآن", "Prefill this request": "قراءة هذا الطلب", "Prefill last request": "قراءة آخر طلب",
    "{count} experts cached": "{count} خبير في الذاكرة", "of {limit} W limit": "من حد قدره {limit} واط",
    "to GPU {rate} MB/s": "إلى البطاقة {rate} ميغابايت/ث", " · idle Gen{gen}": " · الجيل الحالي Gen{gen}",
    "{cores} cores · ": "{cores} نواة · ", "{threads} threads": "{threads} مسار", "write {rate} MB/s": "كتابة {rate} ميغابايت/ث",
    "needs psutil (setup installs it)": "تحتاج إلى psutil (يثبّتها برنامج الإعداد)", "Done": "مكتمل", "Stopped": "متوقف", "Closed": "مغلق",
    "stock": "أصلي", "Copy": "نسخ", "Model": "النموذج", "Engine": "المحرك", "built from source": "مبني من المصدر",
    "Context": "السياق", "KV cache": "ذاكرة KV", "Speculation": "التوليد الاستباقي", "Images": "الصور", "on": "مفعّل", "off": "معطّل",
    "8-bit": "8 بت", "16-bit": "16 بت", "4-bit (Hadamard-rotated)": "4 بت (بتحويل Hadamard)",
    ", all in VRAM": "، كلها في ذاكرة البطاقة", ", streamed: {count} positions per layer in VRAM, the rest in RAM": "، متدفقة: {count} موضع لكل طبقة في ذاكرة البطاقة والبقية في ذاكرة النظام",
    "MTP drafts up to {count} tokens{lookup}": "مسودات MTP حتى {count} رمز{lookup}", ", prompt lookup on": "، البحث في المدخلات مفعّل",
    "not readable (NVML)": "تعذّرت قراءة بيانات البطاقة", "OpenAI base URL": "عنوان OpenAI الأساسي", "Anthropic base URL": "عنوان Anthropic الأساسي",
    "Model name": "اسم النموذج", "Connected": "متصل", "Starting": "جارٍ البدء", "Failed": "فشل", "Waiting": "في الانتظار",
    "{tools} tools · {ready} of {total} servers connected": "{tools} أداة · {ready} من {total} خادم متصل",
    "{tools} tools from {servers}; the model calls them when it decides to": "{tools} أداة من {servers}؛ يستدعيها النموذج عند الحاجة",
    "no server is connected yet (see the Monitor)": "لم يتصل أي خادم بعد (راجع المراقبة)", "{count} tools": "{count} أداة",
    "code": "كود", "Copy code": "نسخ الكود", "You": "أنت", "Copy the answer": "نسخ الإجابة", "Running": "جارٍ التنفيذ", "Not run": "لم تُنفّذ",
    "Arguments": "المعاملات", "(being written)": "(جارٍ الكتابة)", "Result": "النتيجة", " · {count} characters": " · {count} حرف",
    ", cut for the model": "، اختُصرت للنموذج", "Thoughts": "التفكير", "Thought for {seconds} s": "فكّر لمدة {seconds} ث",
    "Shift+Enter: new line": "Shift+Enter: سطر جديد", "the engine reported an error": "أبلغ المحرك عن خطأ", "The request failed": "فشل الطلب",
    " · stopped": " · متوقف", " · projection on": " · الإسقاط مفعّل", " · projection off": " · الإسقاط معطّل",
    "{count} tool call": "{count} استدعاء للأدوات", "{count} tool calls": "{count} استدعاء للأدوات", " · stopped at the limit of {count} tool rounds (mcp.max_rounds)": " · توقف عند حد {count} جولة للأدوات (mcp.max_rounds)",
    "Still writing": "لا تزال الإجابة تُكتب", "Stop the answer first.": "أوقف الإجابة أولًا.", "The last one was cleared.": "مُسحت المحادثة السابقة.",
    "Undo": "تراجع", "Nothing to save yet": "لا يوجد ما يُحفظ بعد", "Pictures are off": "الصور معطّلة", "This model was set up for text only.": "أُعدّ هذا النموذج للنصوص فقط.",
    "Picture too large": "الصورة كبيرة جدًا", "{name} is over 20 MB.": "يتجاوز حجم {name} ‏20 ميغابايت.", "pasted image": "صورة ملصقة",
    "Not a text file": "ليس ملفًا نصيًا", "{name}: attach text files (code, notes, logs, data){pictures}.": "{name}: أرفق ملفات نصية (كود، ملاحظات، سجلات، بيانات){pictures}.",
    " or pictures": " أو صورًا", "File too large": "الملف كبير جدًا", "{name} is over 512 KB.": "يتجاوز حجم {name} ‏512 كيلوبايت.",
    "{name} looks like a binary file.": "يبدو أن {name} ملف ثنائي.", "Remove": "إزالة", "0 · greedy": "0 · اختيار الأكثر احتمالًا",
    "answers right away": "يجيب مباشرة", "short": "مختصر", "medium": "متوسط", "thorough (default)": "مفصّل (الافتراضي)",
    "Sampling saved": "حُفظت إعدادات التوليد", "Other apps (omp, API clients) use these settings from their next request.": "تستخدم التطبيقات الأخرى (omp وتطبيقات API) هذه الإعدادات من الطلب التالي.",
    "Other apps use their own settings again.": "عادت التطبيقات الأخرى إلى إعداداتها الخاصة.", "Saved here, but not for other apps": "حُفظت هنا، لكن تعذّر تطبيقها على التطبيقات الأخرى",
    "Greedy: the same question gives the same answer.": "اختيار الأكثر احتمالًا: السؤال نفسه يعطي الإجابة نفسها.",
    "Since {since}: {requests} requests · {read} prompt tokens read{pSpeed} ({reused} reused) · {output} written{oSpeed}": "منذ {since}: {requests} طلب · قُرئ {read} رمز من المدخلات{pSpeed} ({reused} مُعاد استخدامها) · كُتب {output} رمز{oSpeed}",
    "Local inference": "تشغيل محلي", "Strata API Monitor": "مراقبة API في Strata", "API Monitor": "مراقبة API",
    "Model status and every request, in one place.": "حالة النموذج وجميع الطلبات في مكان واحد.", "Load model": "تحميل النموذج", "Unload model": "تفريغ النموذج",
    "Only needed if the server has a key": "مطلوب إذا كان الخادم يستخدم مفتاحًا فقط", "Connect": "اتصال", "Requests retained": "الطلبات المحفوظة",
    "Waiting for server": "بانتظار الخادم", "Last wall-clock": "مدة آخر طلب", "Includes queue and model loading": "تشمل الانتظار وتحميل النموذج",
    "Last decode speed": "سرعة التوليد الأخيرة", "Engine timing · tokens / second": "توقيت المحرك · رمز / ثانية", "Requests": "الطلبات",
    "Filter requests": "تصفية الطلبات", "Search ID or endpoint": "بحث بالمعرّف أو المسار", "Request status": "حالة الطلب", "All statuses": "جميع الحالات",
    "Active": "نشط", "Completed": "مكتمل", "Errors": "الأخطاء", "Request list": "قائمة الطلبات", "Inspect a request": "فحص طلب",
    "Select a request to see its input, output, and timings.": "اختر طلبًا لعرض مدخلاته ومخرجاته وأوقاته.", "Wall-clock": "المدة الإجمالية",
    "Model load": "تحميل النموذج", "Queue wait": "مدة الانتظار", "First token": "أول رمز", "Prompt tokens": "رموز المدخلات", "Output tokens": "رموز المخرجات",
    "Decode speed": "سرعة التوليد", "Response format": "صيغة الإجابة", "Request content": "محتوى الطلب", "Input": "المدخلات", "Reasoning": "التفكير", "API response": "إجابة API",
    "Last 100 requests kept in memory until restart. Inputs and outputs stay on this machine.": "تُحفظ آخر 100 طلب في الذاكرة حتى إعادة التشغيل. تبقى المدخلات والمخرجات على هذا الجهاز.",
    "No matching requests.": "لا توجد طلبات مطابقة.", "No requests yet. API calls will appear here automatically.": "لا توجد طلبات بعد. تظهر استدعاءات API هنا تلقائيًا.",
    "Request {id}": "الطلب {id}", "stream": "متدفق", "JSON response": "إجابة JSON", "No content.": "لا يوجد محتوى.",
    "Streaming response: see Output and the request error, if any.": "إجابة متدفقة: راجع المخرجات وخطأ الطلب إن وجد.",
    "Raw model output retained for diagnosis. The API returned an error.": "حُفظت مخرجات النموذج الخام للتشخيص. أعادت API خطأً.",
    "Original request body.": "محتوى الطلب الأصلي.", "Model answer. JSON is formatted here for readability.": "إجابة النموذج. نُسّقت JSON هنا لتسهيل القراءة.",
    "Separate reasoning content, when enabled.": "محتوى التفكير المنفصل عند تفعيله.", "API response body.": "محتوى إجابة API.",
    " Monitor capture truncated at 256K characters; the API response was not shortened.": " اختُصرت نسخة المراقبة عند 256 ألف حرف؛ لم تُختصر إجابة API.",
    "Loading…": "جارٍ التحميل…", "Unloading…": "جارٍ التفريغ…", "In use": "قيد الاستخدام", "Loaded": "محمّل", "Unloaded": "غير محمّل",
    "{count} active / queued": "{count} نشط / في الانتظار", "Loads automatically on the next request": "يُحمّل تلقائيًا مع الطلب التالي",
    "No active requests": "لا توجد طلبات نشطة", "Live · updated {time}": "مباشر · حُدّث {time}", "Disconnected · retrying": "انقطع الاتصال · إعادة المحاولة",
    "Copied": "نُسخ", "Clipboard unavailable; select the text to copy it.": "الحافظة غير متاحة؛ حدد النص لنسخه.",
    "completed": "مكتمل", "error": "خطأ", "disconnected": "انقطع الاتصال", "queued": "في الانتظار", "loading": "جارٍ التحميل", "generating": "جارٍ التوليد",
    "reading": "جارٍ القراءة", "text": "نص",
    "{mode} control vector on layers {a}–{b}{direction}. Per chat in Sampling. Its package describes the vector as a refusal-direction projection; measure the speed yourself": "متجه تحكم {mode} على الطبقات {a}–{b}{direction}. يُضبط لكل محادثة من إعدادات التوليد. تصفه الحزمة بإسقاط اتجاه الرفض؛ قِس السرعة بنفسك",
    "Projection": "بالإسقاط", "Additive": "بالإضافة", " (layer {layer}'s direction)": " (باتجاه الطبقة {layer})"
  };
  let language = "en";
  try { if (localStorage.getItem("strata.language") === "ar") language = "ar"; } catch (_) {}
  const locale = () => language === "ar" ? "ar-SA-u-nu-latn" : "en-US";
  function t(key, values = {}) {
    const text = language === "ar" && Object.hasOwn(ar, key) ? ar[key] : key;
    return text.replace(/\{(\w+)\}/g, (match, name) => Object.hasOwn(values, name) ? String(values[name]) : match);
  }
  function direction() {
    document.documentElement.lang = language;
    document.documentElement.dir = language === "ar" ? "rtl" : "ltr";
  }
  function apply() {
    direction();
    document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
    for (const attr of ["title", "placeholder", "aria-label"]) {
      document.querySelectorAll(`[data-i18n-${attr}]`).forEach(el => { el.setAttribute(attr, t(el.getAttribute(`data-i18n-${attr}`))); });
    }
    document.querySelectorAll("[data-language]").forEach(el => { el.value = language; });
  }
  function setLanguage(value, save = true) {
    language = value === "ar" ? "ar" : "en";
    if (save) try { localStorage.setItem("strata.language", language); } catch (_) {}
    apply();
    window.dispatchEvent(new Event("strata-language"));
  }
  direction(); // synchronous, before first paint
  document.addEventListener("DOMContentLoaded", apply);
  document.addEventListener("change", event => {
    if (event.target.matches("[data-language]")) setLanguage(event.target.value);
  });
  window.addEventListener("storage", event => {
    if (event.key === "strata.language") setLanguage(event.newValue, false);
  });
  return {t, locale, apply, setLanguage, get language() { return language; }};
})();
