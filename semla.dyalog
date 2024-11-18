:Namespace SemanticSearch
    ⍝ curl -s -X POST "http://localhost:8000/search" \
    ⍝  -H "Content-Type: application/json" \
    ⍝ -d '{"query": "how do I serialise and compress an apl array?", "k": 5}' | jq .
    ⎕IO←0 ⋄ ⎕ML←1
    hc←⎕SE.SALT.Load'HttpCommand'

    base←'http://localhost:8000/'
    
    semla←{
      (params←⎕NS⍬).(query k)←⍵ 5
      url←base,'search'
      body←(_←hc.GetJSON 'POST' url params '').Data
      body.urls
    }

    ∇ r←List
      r←⎕NS¨1⍴⊂⍬
      r[0].(Name Group Desc Parse)←'Semla' 'SemanticSearch' 'Search phrase' '1S'
    ∇

    ∇ r←Run(Cmd Input)
      r←⍬
      :Select Cmd
      :Case 'Semla'
        r←↑semla (⊃Input.Arguments)
      :Else
        r←⊂'Unrecognised command'
      :EndSelect
    ∇

:EndNamespace
