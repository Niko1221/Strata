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
    "Projection": "بالإسقاط", "Additive": "بالإضافة", " (layer {layer}'s direction)": " (باتجاه الطبقة {layer})",
    "VRAM hit rate": "نسبة إصابة ذاكرة البطاقة",
    "While writing the answer: the share of the experts looked up that were already in VRAM. Experts the GPU read over PCIe (--pcie-frac) are not counted in it; their share of all routed experts is shown after it (+N% PCIe)": "أثناء كتابة الإجابة: نسبة الخبراء الموجودين مسبقًا في ذاكرة البطاقة. لا تشمل الخبراء الذين تقرؤهم البطاقة عبر PCIe (--pcie-frac)؛ تظهر نسبتهم من مجموع الخبراء بعد ذلك (+N% PCIe)",
    "routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed": "الخبراء الذين قرأتهم البطاقة عبر PCIe (--pcie-frac) أو حسبتهم بطاقة أخرى",
    "Conversation cache": "ذاكرة المحادثات المؤقتة",
    "Parked conversations": "المحادثات المحفوظة مؤقتًا",
    "Their memory (RAM)": "ذاكرتها في RAM",
    "Last request": "الطلب الأخير",
    "Reused since start": "المُعاد استخدامه منذ التشغيل",
    "Parked / restored": "المحفوظة / المستعادة",
    "Last switch": "آخر تبديل",
    "just now": "الآن",
    "{count} min ago": "قبل {count} دقيقة",
    "{count} h ago": "قبل {count} ساعة",
    "{reused} of {total} requests reused part of their prompt": "أُعيد استخدام جزء من المدخلات في {reused} من {total} طلبًا",
    " ({pct}% of all prompt tokens)": " ({pct}% من جميع رموز المدخلات)",
    "{event} {count} tokens, {since}": "{event} {count} رمزًا، {since}",
    "Parked": "حُفظت",
    "Restored": "استُعيدت",
    "{reused} of {total} prompt tokens reused": "أُعيد استخدام {reused} من {total} رمزًا من المدخلات",
    "{count} tokens{share}": "{count} رمزًا{share}",
    " · {count} evicted": " · أُزيلت {count}",
    "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full.": "يستعيد الطلب الذي يتابع محادثة محفوظة مؤقتًا حالتها دون قراءتها مجددًا؛ تُزال المحادثة الأقدم عند امتلاء الأماكن أو الذاكرة.",
    "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).": "يحتفظ المحرك بحالة المحادثة الأخيرة، لذلك تقرأ المتابعة المحتوى الجديد فقط. لحفظ عدة محادثات (وكلاء يتناوبون)، أضف \"--conversation-cache-mib\", \"8192\" إلى args في ملف إعداد التشغيل (docs/DETAILS.md).",
    "Model settings": "إعدادات النموذج",
    "Save": "حفظ",
    "default": "الافتراضي",
    "Nothing changed.": "لم يتغير شيء.",
    "Kept in the model's run config for every client. An empty field is the default. They take effect the next time the model starts (close Strata's window and start it again).": "تُحفظ في إعداد تشغيل النموذج لجميع التطبيقات. يعني الحقل الفارغ استخدام القيمة الافتراضية. تسري عند تشغيل النموذج مجددًا (أغلق نافذة Strata وأعد تشغيله).",
    "Saved ({keys}); the earlier file is {file}.bak. Start the model again to use it.": "حُفظت ({keys})؛ الملف السابق هو {file}.bak. أعد تشغيل النموذج لاستخدامها.",
    "Not saved": "لم تُحفظ",
    "Default temperature for requests that send none (0 = greedy, the default without a sampling block)": "درجة الحرارة الافتراضية للطلبات التي لا تحددها (0 = اختيار حتمي، وهو الافتراضي دون إعدادات توليد)",
    "Default top_p for requests that send none": "قيمة top_p الافتراضية للطلبات التي لا تحددها",
    "Default top_k for requests that send none (1-64)": "قيمة top_k الافتراضية للطلبات التي لا تحددها (1–64)",
    "Default min_p for requests that send none": "قيمة min_p الافتراضية للطلبات التي لا تحددها",
    "Cap the thinking of every request at this many tokens (0 or empty: no cap)": "الحد الأقصى لرموز التفكير لكل طلب (0 أو فارغ: دون حد)",
    "Shorten a max_tokens that does not fit the context instead of answering 400": "تقليل max_tokens إذا تجاوز سعة السياق بدلًا من إرجاع الخطأ 400",
    "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)": "طلبات Anthropic التي لا تطلب التفكير: التفكير وفق النموذج (model) أو عند الطلب فقط (on_request)",
    "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)": "موضع مستوى التفكير غير الافتراضي: البداية (الافتراضي) أو النهاية (يحافظ على الذاكرة المؤقتة عند تغييره)",
    "Other model names the server lists and answers to (comma-separated)": "أسماء أخرى للنموذج يعرضها الخادم ويستجيب لها (مفصولة بفواصل)",
    "Unload the model after this many seconds without requests (0 or empty: never)": "تفريغ النموذج بعد هذا العدد من الثواني دون طلبات (0 أو فارغ: لا يُفرّغ)",
    "Start without loading the model; the first request loads it (text only)": "البدء دون تحميل النموذج؛ يحمّله الطلب الأول (للنص فقط)",
    "End a request when the engine says nothing for this long (default 300 s, 0 = wait)": "إنهاء الطلب إذا بقي المحرك صامتًا لهذه المدة (الافتراضي 300 ثانية، 0 = الانتظار)",
    "Keep the last 100 requests' prompts and answers in memory for /api-monitor": "حفظ مدخلات وإجابات آخر 100 طلب في الذاكرة لمراقبة API",
    "Open the chat page in the browser when the model is ready": "فتح صفحة المحادثة في المتصفح عند جاهزية النموذج",
    "VRAM in MiB the engine leaves free for other programs (engine default 700)": "ذاكرة البطاقة بوحدة MiB التي يتركها المحرك للبرامج الأخرى (الافتراضي 700)",
    "not available": "غير متاح",
    "not available on Windows AMD yet": "غير متاح حاليًا لبطاقات AMD على Windows",
    "no GPU load or VRAM readings for AMD cards on Windows yet (Linux reads them from the amdgpu driver); the engine's own VRAM figures are in its log": "لا تتوفر حاليًا قراءات حمل البطاقة أو ذاكرة VRAM لبطاقات AMD على Windows (يقرأها Linux من برنامج تشغيل amdgpu)؛ تظهر أرقام ذاكرة المحرك في سجله",
    "Prefill t/s": "قراءة المدخلات رمز/ث",
    "Decode t/s": "التوليد رمز/ث",
    "New prompt tokens per second, excluding reused cache tokens (engine prompt time)": "عدد رموز المدخلات الجديدة في الثانية، باستثناء الرموز المُعاد استخدامها من الذاكرة المؤقتة (زمن قراءة المحرك)"
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
