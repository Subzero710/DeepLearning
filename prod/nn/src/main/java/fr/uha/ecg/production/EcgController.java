package fr.uha.ecg.production;

import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/api")
public class EcgController {
    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ia.base-url:${IA_BASE_URL:http://ia:80}}")
    private String iaBaseUrl;

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return exchange(iaBaseUrl + "/health", HttpMethod.GET, HttpEntity.EMPTY);
    }

    @PostMapping("/config")
    public ResponseEntity<String> config() {
        return exchange(iaBaseUrl + "/config", HttpMethod.POST, HttpEntity.EMPTY);
    }

    @PostMapping(value = "/predict", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> predict(@RequestBody Map<String, Object> body) {
        return postJson(iaBaseUrl + "/predict", body);
    }

    @PostMapping(value = "/batch", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> batch(@RequestBody Map<String, Object> body) {
        return postJson(iaBaseUrl + "/batch", body);
    }

    private ResponseEntity<String> postJson(String url, Map<String, Object> body) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);
        return exchange(url, HttpMethod.POST, request);
    }

    private ResponseEntity<String> exchange(String url, HttpMethod method, HttpEntity<?> request) {
        try {
            return restTemplate.exchange(url, method, request, String.class);
        } catch (HttpStatusCodeException exc) {
            return ResponseEntity.status(exc.getStatusCode())
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(exc.getResponseBodyAsString());
        }
    }
}
