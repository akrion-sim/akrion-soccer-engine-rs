# Soccer thread model

The live server has worker threads, but a single live match does not have two
tick loops running at once. The game state for each live match is behind one
`Arc<Mutex<SoccerRealtimeSession>>`; whichever HTTP worker handles `/api/step`
must hold that session lock while it advances ticks.

```mermaid
flowchart TB
  subgraph LIVE["Live server process: main_soccer_live / run_live_soccer_server"]
    ACCEPT["accept loop<br/>TcpListener::incoming"]
    QUEUE["bounded stream queue<br/>mpsc::sync_channel"]
    WORKERS["HTTP worker pool<br/>soccer-live-http-worker-N<br/>default 4, max 32"]
    REGISTRY["LiveGameRegistry<br/>default game + ?game=id sessions"]
  end

  subgraph GAME_A["One live game session"]
    SESSION_A["Arc&lt;Mutex&lt;SoccerRealtimeSession&gt;&gt;<br/>per-game session lock"]
    STEP_A["/api/step<br/>step_for_live_http"]
    TICK_A["SoccerMatch::run_time_step<br/>single tick executor while locked"]
    INPUT_Q_A["SharedHumanInputs<br/>input frames drained by the next step"]
    CTRL_A["human controller threads<br/>one mailbox/condvar per slot"]
  end

  subgraph GAME_B["Another ?game=id session"]
    SESSION_B["separate Arc&lt;Mutex&lt;SoccerRealtimeSession&gt;&gt;"]
    STEP_B["/api/step for that game"]
    TICK_B["independent tick executor"]
  end

  subgraph SIDE["Side workers that do not directly run live ticks"]
    INPUT_POST["/api/input<br/>HumanControllerInputRouter"]
    NEURAL["optional neural learning worker<br/>threaded backend only"]
    PG_REFRESH["optional Postgres policy refresh<br/>postgres-persistence feature"]
    ADMIN["short-lived restart/signal helper threads"]
  end

  ACCEPT --> QUEUE --> WORKERS --> REGISTRY
  REGISTRY --> SESSION_A
  REGISTRY --> SESSION_B

  WORKERS --> STEP_A --> SESSION_A --> TICK_A
  WORKERS --> STEP_B --> SESSION_B --> TICK_B

  INPUT_POST --> CTRL_A --> INPUT_Q_A --> STEP_A
  NEURAL -->|"batch results / snapshots"| SESSION_A
  PG_REFRESH -->|"short lock-held policy swap"| SESSION_A
  ADMIN -.-> LIVE

  classDef live fill:#1f3a5f,stroke:#5b9bd5,color:#fff;
  classDef game fill:#1f5f3a,stroke:#5bd59b,color:#fff;
  classDef side fill:#5f4a1f,stroke:#d5b55b,color:#fff;
  class ACCEPT,QUEUE,WORKERS,REGISTRY live;
  class SESSION_A,STEP_A,TICK_A,INPUT_Q_A,CTRL_A,SESSION_B,STEP_B,TICK_B game;
  class INPUT_POST,NEURAL,PG_REFRESH,ADMIN side;
```

## Live-game contract

- There are multiple live-server threads: the accept loop, HTTP workers, human
  controller threads, and optional side workers.
- A single game session advances ticks under one session mutex. Concurrent
  `/api/step` requests for that same game serialize on that lock.
- Named `?game=id` sessions are independent. Two different game ids may advance
  in parallel, because they have different session mutexes.
- `/api/input` can enqueue controller frames without stepping the match. Those
  inputs are consumed by the next locked tick batch.
- The runtime exposes this in `liveHttp`: `simulationStepModel` is
  `per-game-session-mutex`, with `gameTicksSingleThreadedPerSession=true` and
  `concurrentStepRequestsSerialized=true`.

## Learning and tournament workers

```mermaid
flowchart LR
  subgraph CONT["Continuous learner<br/>main_soccer_learning_run"]
    CONT_DEFAULT["default parallel_games=1"]
    CONT_ENV["explicit SOCCER_PARALLEL_GAMES=N"]
    CONT_POOL["SoccerLearningWorkerPool<br/>only fans out when N &gt; 1"]
  end

  subgraph QUEUE["Queue farm<br/>main_soccer_learning_queue"]
    QUEUE_ENV["SOCCER_QUEUE_PARALLEL_GAMES=N<br/>or SOCCER_PARALLEL_GAMES fallback"]
    QUEUE_POOL["queue self-play workers"]
    QUEUE_PG["async Postgres completed-run writer"]
  end

  subgraph TOUR["Tournament runner<br/>main_soccer_tournament_run"]
    TOUR_ENV["SOCCER_TOURNAMENT_THREADS=N"]
    TOUR_SCOPE["std::thread::scope<br/>parallel matches per chunk"]
  end

  CONT_DEFAULT --> CONT_POOL
  CONT_ENV --> CONT_POOL
  QUEUE_ENV --> QUEUE_POOL --> QUEUE_PG
  TOUR_ENV --> TOUR_SCOPE

  classDef cont fill:#1f5f3a,stroke:#5bd59b,color:#fff;
  classDef queue fill:#3a1f5f,stroke:#9b5bd5,color:#fff;
  classDef tour fill:#5f1f3a,stroke:#d55b9b,color:#fff;
  class CONT_DEFAULT,CONT_ENV,CONT_POOL cont;
  class QUEUE_ENV,QUEUE_POOL,QUEUE_PG queue;
  class TOUR_ENV,TOUR_SCOPE tour;
```

## Runner contract

- `main_soccer_learning_run` is the deterministic continuous lane. Its default is
  one game; extra parallelism must be explicit.
- Scale-out belongs in `main_soccer_learning_queue` and tournament runs, where
  parallel workers are the point of the runner.
- Tournament parallelism is scoped and deterministic in result ordering: matches
  may run concurrently, then outcomes are collected into the original order.
