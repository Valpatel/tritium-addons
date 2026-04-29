// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Matrix communication addon tab

(function() {
    function reg() {
        if (!window._tritiumEventBus) { setTimeout(reg, 500); return; }
        window._tritiumEventBus.emit("panel:register-tab", {
            container: "comms-container",
            id: "matrix-tab",
            title: "MATRIX",
            addon: true,
            create: function(el) {
                el.innerHTML = "<div style=\"padding:8px;font-family:monospace;font-size:11px;color:#ccc\">"
                    + "<div style=\"color:#0dbd8b;font-size:12px;margin-bottom:8px\">MATRIX</div>"
                    + "<p style=\"color:#555;font-size:10px;margin-bottom:10px\">Matrix/Element federated messaging</p>"
                    + "<div style=\"color:#888;margin-bottom:6px;font-size:10px\">CONFIGURATION</div>"
                    + "<div class=\"cf-row\"><label class=\"cf-label\">HOMESERVER</label><input type=\"text\" class=\"cf-input\" data-key=\"homeserver\" placeholder=\"Homeserver URL\"></div><div class=\"cf-row\"><label class=\"cf-label\">USER_ID</label><input type=\"text\" class=\"cf-input\" data-key=\"user_id\" placeholder=\"Bot user ID\"></div><div class=\"cf-row\"><label class=\"cf-label\">ACCESS_TOKEN</label><input type=\"text\" class=\"cf-input\" data-key=\"access_token\" placeholder=\"Access token\"></div><div class=\"cf-row\"><label class=\"cf-label\">ROOM_ID</label><input type=\"text\" class=\"cf-input\" data-key=\"room_id\" placeholder=\"Alert room\"></div>"
                    + "<div style=\"margin-top:10px;display:flex;gap:4px\">"
                    + "<button class=\"cf-btn\" data-action=\"save\" style=\"color:#0dbd8b;border-color:#0dbd8b\">SAVE</button>"
                    + "<button class=\"cf-btn\" data-action=\"test\" style=\"color:#888;border-color:#333\">TEST</button>"
                    + "<button class=\"cf-btn\" data-action=\"enable\" style=\"color:#05ffa1;border-color:#05ffa1\">ENABLE</button>"
                    + "</div>"
                    + "<div style=\"margin-top:8px;font-size:10px;color:#666\" data-bind=\"feedback\"></div>"
                    + "</div>"
                    + "<style>.cf-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0}"
                    + ".cf-label{color:#666;font-size:10px;min-width:100px}"
                    + ".cf-input{background:#0a0a12;border:1px solid #1a1a2e;color:#ccc;padding:2px 6px;font-family:inherit;font-size:10px;flex:1;margin-left:8px}"
                    + ".cf-input[type=checkbox]{flex:none;width:14px;height:14px}"
                    + ".cf-btn{background:#0a0a12;border:1px solid;padding:3px 10px;font-family:inherit;font-size:10px;cursor:pointer}"
                    + ".cf-btn:hover{filter:brightness(1.3)}</style>";
                var save = el.querySelector("[data-action=save]");
                if (save) save.onclick = function() {
                    var c = {};
                    el.querySelectorAll(".cf-input").forEach(function(i) { c[i.dataset.key] = i.type === "checkbox" ? i.checked : i.value; });
                    fetch("/api/comms/matrix/config", {method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(c)})
                        .then(function(){ var f=el.querySelector("[data-bind=feedback]"); if(f){f.textContent="Saved";f.style.color="#05ffa1";} });
                };
                var test = el.querySelector("[data-action=test]");
                if (test) test.onclick = function() {
                    fetch("/api/comms/matrix/test",{method:"POST"}).then(function(r){return r.json();}).then(function(d){
                        var f=el.querySelector("[data-bind=feedback]"); if(f){f.textContent=d.success?"OK":"Not configured";f.style.color=d.success?"#05ffa1":"#ff2a6d";}
                    });
                };
                var en = el.querySelector("[data-action=enable]");
                if (en) en.onclick = function() {
                    fetch("/api/comms/matrix/config",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({enabled:true})})
                        .then(function(){ var f=el.querySelector("[data-bind=feedback]"); if(f){f.textContent="Enabled";f.style.color="#05ffa1";} });
                };
            }
        });
    }
    reg();
})();
